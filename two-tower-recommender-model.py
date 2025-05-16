# Databricks notebook source
# MAGIC %md # Two Tower (using TorchRec + TorchDistributor + StreamingDataset)
# MAGIC
# MAGIC This notebook illustrates how to create a distributed Two Tower recommendation model. This notebook was tested on `g5dn.12xlarge` instances (one instance as the driver, one instance as the worker) using the Databricks Runtime for ML 16.4 LTS. For more insight into the Two Tower recommendation model, you can view the following resources:
# MAGIC - Hopsworks definition: https://www.hopsworks.ai/dictionary/two-tower-embedding-model
# MAGIC - TorchRec's training implementation: https://github.com/pytorch/torchrec/blob/main/examples/retrieval/two_tower_train.py#L75
# MAGIC
# MAGIC **Note: Where you see `# TODO` in this notebook, you must enter custom code to ensure that the notebook runs successfully.**
# MAGIC
# MAGIC ## Requirements
# MAGIC This notebook requires <DBR> 16.4 LTS ML, along with the libraries listed in the requirements file.
# MAGIC
# MAGIC This notebook is derived from the [two-tower recommender system example](https://docs.databricks.com/aws/en/notebooks/source/machine-learning/two-tower-recommender-model.html) available [in our documentation](https://docs.databricks.com/aws/en/machine-learning/train-recommender-models), but updated to work with 16.4 LTS ML along with a few other fixes. Updates have been highlighted in the comments.

# COMMAND ----------

# MAGIC %md ## 1. Saving "Learning From Sets of Items" Data in UC Volumes in the MDS (Mosaic Data Shard) format
# MAGIC
# MAGIC This notebook uses the small sample of 100k ratings from ["Learning From Sets of Items"](https://grouplens.org/datasets/learning-from-sets-of-items-2019/). In this section you preprocess it and save it to a Volume in Unity Catalog.

# COMMAND ----------

# MAGIC %md ### 1.1. Downloading the Dataset
# MAGIC
# MAGIC Download the dataframe from `https://grouplens.org/datasets/learning-from-sets-of-items-2019/` to `/databricks/driver` and then save the data to a UC Table. The "Learning from Sets of Items" dataset has the Creative Commons 4.0 license.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "users")
dbutils.widgets.text("schema_name", "corey_abshire")
dbutils.widgets.text("ratings_table_name", "learning_from_sets_data")
dbutils.widgets.text("volume_name", "movielens")

catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")
ratings_table_name = dbutils.widgets.get("ratings_table_name")
volume_name = dbutils.widgets.get("volume_name")

print(f"catalog_name: {catalog_name}")
print(f"schema_name: {schema_name}")
print(f"ratings_table_name: {ratings_table_name}")
print(f"volume_name: {volume_name}")

assert catalog_name != "", "catalog_name is required"
assert schema_name != "", "schema_name is required"
assert ratings_table_name != "", "ratings_table_name is required"
assert volume_name != "", "volume_name is required"

ratings_table_name = f"{catalog_name}.{schema_name}.{ratings_table_name}"
mds_volume_path = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/{ratings_table_name}"

print(f"ratings_table_name: {ratings_table_name}")
print(f"mds_volume_path: {mds_volume_path}")

# COMMAND ----------

# MAGIC %sh
# MAGIC # Update: Make this idempotent by testing whether file exists before downloading.
# MAGIC #         (otherwise by default wget will error out). Also use variables.
# MAGIC
# MAGIC # Define variables for paths
# MAGIC filename="learning-from-sets-2019.zip"
# MAGIC url="https://files.grouplens.org/datasets/learning-from-sets-2019/$filename"
# MAGIC zip_path="/databricks/driver/$filename"
# MAGIC unzip_dir="/databricks/driver/"
# MAGIC
# MAGIC # Check if the file exists before downloading
# MAGIC if [ ! -f "$zip_path" ]; then
# MAGIC     wget url -O "$zip_path"
# MAGIC fi
# MAGIC
# MAGIC # Unzip the file (overwrite if exists)
# MAGIC unzip -o "$zip_path" -d "$unzip_dir"

# COMMAND ----------

import pandas as pd

# Load the CSV file into a pandas DataFrame (since the data is stored on local machine)
df = pd.read_csv("/databricks/driver/learning-from-sets-2019/item_ratings.csv")

# Create a Spark DataFrame from the pandas DataFrame and save it to UC
spark_df = spark.createDataFrame(df)

# Update: Pulled up the table name to a configuration parameter and made
#         creation mode as ignore to help with idempotent runs.
# TODO: Update this with a path in UC for where this data should be saved
spark_df.write.mode("ignore").saveAsTable(ratings_table_name)

spark_df = spark.table(ratings_table_name)

# COMMAND ----------

# MAGIC %md ### 1.2. Reading the Dataset from UC
# MAGIC
# MAGIC The original dataset contains 500k data points. This example uses a sample of 100k data points from the dataset.

# COMMAND ----------

# TODO: Update this with the path in UC where this data is saved
spark_df = spark.table(ratings_table_name)
print(f"Dataset size: {spark_df.count()}")
display(spark_df)

# COMMAND ----------

# Order by userId and movieId (this allows you to get a better representation of movieIds and userIds for the dataset)

# Update: There are only ~500k ratings in this file anyway, so I set it to 1M to get them all. With the other changes
#         it should still run fine, and is still a relatively small dataset.
ordered_df = spark_df.orderBy("userId", "movieId").limit(1_000_000)

print(f"Updated Dataset Size: {ordered_df.count()}")
# Show the result
display(ordered_df)

# COMMAND ----------

from pyspark.sql.functions import countDistinct

# Get the total number of data points
print("Total # of data points:", ordered_df.count())

# Get the total number of users
total_users = ordered_df.select(countDistinct("userId")).collect()[0][0]
print(f"Total # of users: {total_users}")

# Get the total number of movies
total_movies = ordered_df.select(countDistinct("movieId")).collect()[0][0]
print(f"Total # of movies: {total_movies}")

# COMMAND ----------

# MAGIC %md ### 1.3. Preprocessing and Cleaning the Data
# MAGIC
# MAGIC The first step is to convert the hashes (in string format) of each user to an integer using the StringIndexer.
# MAGIC
# MAGIC The Two Tower Model provided by TorchRec [here](https://github.com/pytorch/torchrec/blob/main/examples/retrieval/two_tower_train.py#L4) requires a binary label. The code in this section converts all ratings less than the mean to `0` and all values greater than the mean to `1`. For your own use case, you can modify the training task described [here](https://github.com/pytorch/torchrec/blob/main/examples/retrieval/modules/two_tower.py#L117) to use MSELoss instead (which can scale to ratings from 0 -> 5).

# COMMAND ----------

from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import LongType

string_indexer = StringIndexer(inputCol="userId", outputCol="userId_index")
indexed_df = string_indexer.fit(ordered_df).transform(ordered_df)
indexed_df = indexed_df.withColumn("userId_index", indexed_df["userId_index"].cast(LongType()))
indexed_df = indexed_df.withColumn("userId", indexed_df["userId_index"]).drop("userId_index")

display(indexed_df)

# COMMAND ----------

from pyspark.sql import functions as F

# Select only the userId, movieId, and ratings columns
relevant_df = indexed_df.select('userId', 'movieId', 'rating')

# Calculate the mean of the ratings column
ratings_mean = relevant_df.groupBy().avg('rating').collect()[0][0]
print(f"Mean rating: {ratings_mean}")

# Modify all ratings less than the mean to 0 and greater than the mean to 1 and using a UDF to apply the transformation
modify_rating_udf = F.udf(lambda x: 0 if x < ratings_mean else 1, 'int')
relevant_df = relevant_df.withColumn('rating', modify_rating_udf('rating'))

# Rename rating to label
relevant_df = relevant_df.withColumnRenamed('rating', 'label')

# Displaying the dataframe
display(relevant_df)

# COMMAND ----------

# MAGIC %md ### 1.4. Saving to MDS Format within UC Volumes
# MAGIC
# MAGIC In this step, you convert the data to MDS to allow for efficient train/validation/test splitting and then save it to a UC Volume.
# MAGIC
# MAGIC View the Mosaic Streaming guide here for more details:
# MAGIC 1. General details: https://docs.mosaicml.com/projects/streaming/en/stable/
# MAGIC 2. Main concepts: https://docs.mosaicml.com/projects/streaming/en/stable/getting_started/main_concepts.html#dataset-conversion
# MAGIC 2. `dataframeToMDS` details: https://docs.mosaicml.com/projects/streaming/en/stable/preparing_datasets/spark_dataframe_to_mds.html

# COMMAND ----------

# Split the dataframe into train, test, and validation sets
train_df, validation_df, test_df = relevant_df.randomSplit([0.7, 0.2, 0.1], seed=42)

# Show the count of each split to verify the distribution
print(f"Training Dataset Count: {train_df.count()}")
print(f"Validation Dataset Count: {validation_df.count()}")
print(f"Test Dataset Count: {test_df.count()}")

# COMMAND ----------

from streaming import StreamingDataset
from streaming.base.converters import dataframe_to_mds
from streaming.base import MDSWriter
from shutil import rmtree
import os
from tqdm import tqdm

# Parameters required for saving data in MDS format
cols = ["userId", "movieId"]
cat_dict = { key: 'int64' for key in cols }

# Update: 'int' is no longer acceptable in later versions of streaming, so just had
#         to change this to int32
label_dict = { 'label' : 'int32' }
columns = {**label_dict, **cat_dict}

compression = 'zstd:7'

# Update: Just modified this to pull the mds_volume_path up to the top
# TODO: Update this with the path in UC where this data is saved
output_dir_train = os.path.join(mds_volume_path, "mds_train")
output_dir_validation = os.path.join(mds_volume_path, "mds_validation")
output_dir_test = os.path.join(mds_volume_path, "mds_test")

# Save the training data using the `dataframe_to_mds` function, which divides the dataframe
# into `num_workers` parts and merges the `index.json` from each part into one in a parent directory.
def save_data(df, output_path, label, num_workers=40):
    # Update: Add a guard to help with notebook idempotency
    if os.path.exists(output_path):
        print(f"Skipping since {output_path} already exists")
        return
    print(f"Saving {label} data to: {output_path}")
    mds_kwargs = {'out': output_path, 'columns': columns, 'compression': compression}
    dataframe_to_mds(df.repartition(num_workers), merge_index=True, mds_kwargs=mds_kwargs)

save_data(train_df, output_dir_train, 'train')
save_data(validation_df, output_dir_validation, 'validation')
save_data(test_df, output_dir_test, 'test')

# COMMAND ----------

# MAGIC %md ## 2. Helper Functions for Recommendation Dataloading
# MAGIC
# MAGIC In this section, you install the necessary libraries, add imports, and add some relevant helper functions to train the model.

# COMMAND ----------

# MAGIC %md ### 2.1. Installs and Imports

# COMMAND ----------

# Update: Commenting these out, as 16.4 + the cluster libs mentioned in requirements.txt should be sufficient.

# %pip install -q --upgrade --no-deps --force-reinstall torch==2.2.2 torchvision==0.17.2 torchrec==0.6.0 fbgemm-gpu==0.6.0 --index-url https://download.pytorch.org/whl/cu118
# %pip install torchmetrics==1.0.3 iopath==0.1.10 pyre_extensions==0.0.32 mosaicml-streaming==0.7.5 
# dbutils.library.restartPython()

# COMMAND ----------

import os
from typing import List, Optional
from streaming import StreamingDataset, StreamingDataLoader

import torch
import torchmetrics as metrics
from torch import distributed as dist
from torch.distributed.optim import (
    _apply_optimizer_in_backward as apply_optimizer_in_backward,
)
from torch.utils.data import DataLoader
from torch import nn
from torchrec import inference as trec_infer
from torchrec.distributed import TrainPipelineSparseDist
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.inference.state_dict_transform import (
    state_dict_gather,
    state_dict_to_device,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.optim.keyed import KeyedOptimizerWrapper
from torchrec.optim.rowwise_adagrad import RowWiseAdagrad
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.datasets.utils import Batch
from torch.distributed._sharded_tensor import ShardedTensor

from collections import defaultdict
from functools import partial
import mlflow

from typing import Tuple, List, Optional
from torchrec.modules.mlp import MLP

from pyspark.ml.torch.distributor import TorchDistributor
from tqdm import tqdm
import torchmetrics as metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2. Helper functions for Converting to Pipelineable DataType
# MAGIC
# MAGIC Using TorchRec pipelines requires a pipelineable data type (which is `Batch` in this case). In this step, you create a helper function that takes each batch from the StreamingDataset and passes it through a transformation function to convert it into a pipelineable batch.
# MAGIC
# MAGIC For further context, see https://github.com/pytorch/torchrec/blob/main/torchrec/datasets/utils.py#L28.

# COMMAND ----------

# Update: Added some logic to automatically get the embedding sizes.
#         Note that this assumes we're using the full ratings dataset,
#         so you could optionally adjust this if using less. (though it
#         should still work as it is OK if the table is oversized).

ratings_df = spark.table(ratings_table_name)
user_count = ratings_df.select("userId").distinct().count()
movie_count = ratings_df.select("movieId").distinct().count()
print(f"User count: {user_count}")
print(f"Movie count: {movie_count}")

# COMMAND ----------

# Update: Because we're dynamically computing the sizes, this TODO is no longer needed.
# TODO: This is from earlier outputs (section 1.2, cell 2); if another dataset is being used, these values need to be updated
cat_cols = ["userId", "movieId"]
emb_counts = [user_count, movie_count]

def transform_to_torchrec_batch(batch, num_embeddings_per_feature: Optional[List[int]] = None) -> Batch:
    kjt_values: List[int] = []
    kjt_lengths: List[int] = []
    for col_idx, col_name in enumerate(cat_cols):
        values = batch[col_name]
        for value in values:
            if value:
                kjt_values.append(
                    value % num_embeddings_per_feature[col_idx]
                )
                kjt_lengths.append(1)
            else:
                kjt_lengths.append(0)

    sparse_features = KeyedJaggedTensor.from_lengths_sync(
        cat_cols,
        torch.tensor(kjt_values),
        torch.tensor(kjt_lengths, dtype=torch.int32),
    )
    labels = torch.tensor(batch["label"], dtype=torch.int32)
    assert isinstance(labels, torch.Tensor)

    return Batch(
        dense_features=torch.zeros(1),
        sparse_features=sparse_features,
        labels=labels,
    )

transform_partial = partial(transform_to_torchrec_batch, num_embeddings_per_feature=emb_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3. Helper Function for DataLoading using Mosaic's StreamingDataset
# MAGIC
# MAGIC This utilizes Mosaic's StreamingDataset and Mosaic's StreamingDataLoader for efficient data loading. For more information, view this [documentation](https://docs.mosaicml.com/projects/streaming/en/stable/distributed_training/fast_resumption.html#saving-and-loading-state). 

# COMMAND ----------

# Update: StreamingDataset works much better if both a remote path and a local
#         path are provided, so we do that here. We also bump up the number of
#         workers, up the prefetch factor, and use pinned memory.

# def get_dataloader_with_mosaic(path, batch_size, label):
#     print(f"Getting {label} data from UC Volumes")
#     dataset = StreamingDataset(local=path, shuffle=True, batch_size=batch_size)
#     return StreamingDataLoader(dataset, batch_size=batch_size)

from streaming.base.util import clean_stale_shared_memory

def get_dataloader_with_mosaic(remote_path, local_path, batch_size):
    print(f"Getting data from {remote_path}")
    clean_stale_shared_memory()
    dataset = StreamingDataset(remote=remote_path, local=local_path, shuffle=True, batch_size=batch_size)
    return StreamingDataLoader(dataset, batch_size=batch_size, num_workers=8, prefetch_factor=2, pin_memory=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Creating the Relevant TorchRec code for Training
# MAGIC
# MAGIC This section contains all of the training and evaluation code.

# COMMAND ----------

# MAGIC %md ### 3.1. Two Tower Model Definition
# MAGIC
# MAGIC This is taken directly from the [torchrec example's page](https://sourcegraph.com/github.com/pytorch/torchrec@2d62bdef24d144eaabeb0b8aa9376ded4a89e9ee/-/blob/examples/retrieval/modules/two_tower.py?L38:7-38:15). Note that the loss is the Binary Cross Entropy loss, which requires labels to be within the values {0, 1}.

# COMMAND ----------

import torch.nn.functional as F

class TwoTower(nn.Module):
    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        layer_sizes: List[int],
        device: Optional[torch.device] = None
    ) -> None:
        super().__init__()

        assert len(embedding_bag_collection.embedding_bag_configs()) == 2, "Expected two EmbeddingBags in the two tower model"
        assert embedding_bag_collection.embedding_bag_configs()[0].embedding_dim == embedding_bag_collection.embedding_bag_configs()[1].embedding_dim, "Both EmbeddingBagConfigs must have the same dimension"

        embedding_dim = embedding_bag_collection.embedding_bag_configs()[0].embedding_dim
        self._feature_names_query: List[str] = embedding_bag_collection.embedding_bag_configs()[0].feature_names
        self._candidate_feature_names: List[str] = embedding_bag_collection.embedding_bag_configs()[1].feature_names
        self.ebc = embedding_bag_collection
        self.query_proj = MLP(in_size=embedding_dim, layer_sizes=layer_sizes, device=device)
        self.candidate_proj = MLP(in_size=embedding_dim, layer_sizes=layer_sizes, device=device)

    def forward(self, kjt: KeyedJaggedTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled_embeddings = self.ebc(kjt)
        query_embedding: torch.Tensor = self.query_proj(
            torch.cat(
                [pooled_embeddings[feature] for feature in self._feature_names_query],
                dim=1,
            )
        )
        candidate_embedding: torch.Tensor = self.candidate_proj(
            torch.cat(
                [
                    pooled_embeddings[feature]
                    for feature in self._candidate_feature_names
                ],
                dim=1,
            )
        )
        return query_embedding, candidate_embedding


class TwoTowerTrainTask(nn.Module):
    def __init__(self, two_tower: TwoTower) -> None:
        super().__init__()
        self.two_tower = two_tower
        # The BCEWithLogitsLoss combines a sigmoid layer and binary cross entropy loss
        self.loss_fn: nn.Module = nn.BCEWithLogitsLoss()

    def forward(self, batch: Batch) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        query_embedding, candidate_embedding = self.two_tower(batch.sparse_features)
        logits = (query_embedding * candidate_embedding).sum(dim=1).squeeze()
        loss = self.loss_fn(logits, batch.labels.float())
        return loss, (loss.detach(), logits.detach(), batch.labels.detach())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2. Base Dataclass for Training inputs
# MAGIC
# MAGIC Feel free to modify any of the variables mentioned here, but note that the first layer for `layer_sizes` should be equivalent to `embedding_dim`.

# COMMAND ----------

from dataclasses import dataclass, field
import itertools 

# TODO: Update these values for training as needed
@dataclass
class Args:
    """
    Training arguments.
    """
    epochs: int = 3  # Training for one Epoch
    embedding_dim: int = 128  # Embedding dimension is 128
    layer_sizes: List[int] = field(default_factory=lambda: [128, 64]) # The layers for the two tower model are 128, 64 (with the final embedding size for the outputs being 64)
    learning_rate: float = 0.01
    batch_size: int = 1024 # Set a larger batch size due to the large size of dataset
    print_sharding_plan: bool = True
    print_lr: bool = False  # Optional, prints the learning rate at each iteration step
    validation_freq: int = None  # Optional, determines how often during training you want to run validation (# of training steps)
    limit_train_batches: int = None  # Optional, limits the number of training batches
    limit_val_batches: int = None  # Optional, limits the number of validation batches
    limit_test_batches: int = None  # Optional, limits the number of test batches

# Store the results in mlflow
def get_relevant_fields(args, cat_cols, emb_counts):
    fields_to_save = ["epochs", "embedding_dim", "layer_sizes", "learning_rate", "batch_size"]
    result = { key: getattr(args, key) for key in fields_to_save }
    # add dense cols
    result["cat_cols"] = cat_cols
    result["emb_counts"] = emb_counts
    return result

# COMMAND ----------

# MAGIC %md ### 3.3. Training and Evaluation Helper Functions

# COMMAND ----------

def batched(it, n):
    assert n >= 1
    for x in it:
        yield itertools.chain((x,), itertools.islice(it, n - 1))

# COMMAND ----------

# MAGIC %md #### 3.3.1. Helper Functions for Distributed Model Saving

# COMMAND ----------

# Two Tower and TorchRec use special tensors called ShardedTensors.
# This code localizes them and puts them in the same rank that is saved to MLflow.
def gather_and_get_state_dict(model):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    state_dict = model.state_dict()
    gathered_state_dict = {}

    # Iterate over all items in the state_dict
    for fqn, tensor in state_dict.items():
        if isinstance(tensor, ShardedTensor):
            # Collect all shards of the tensor across ranks
            full_tensor = None
            if rank == 0:
                full_tensor = torch.zeros(tensor.size()).to(tensor.device)
            tensor.gather(0, full_tensor)
            if rank == 0:
                gathered_state_dict[fqn] = full_tensor
        else:
            # Directly add non-sharded tensors to the new state_dict
            if rank == 0:
                gathered_state_dict[fqn] = tensor

    return gathered_state_dict

def log_state_dict_to_mlflow(model, artifact_path) -> None:
    # All ranks participate in gathering
    state_dict = gather_and_get_state_dict(model)
    # Only rank 0 logs the state dictionary
    if dist.get_rank() == 0 and state_dict:
        mlflow.pytorch.log_state_dict(state_dict, artifact_path=artifact_path)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.3.2. Helper Functions for Distributed Model Training and Evaluation

# COMMAND ----------

import torchmetrics as metrics

def evaluate(
    limit_batches: Optional[int],
    pipeline: TrainPipelineSparseDist,
    eval_dataloader: DataLoader,
    stage: str) -> Tuple[float, float]:
    """
    Evaluates model. Computes and prints AUROC and average loss. Helper function for train_val_test.

    Args:
        limit_batches (Optional[int]): Limits the dataloader to the first `limit_batches` batches.
        pipeline (TrainPipelineSparseDist): data pipeline.
        eval_dataloader (DataLoader): Dataloader for either the validation set or test set.
        stage (str): "val" or "test".

    Returns:
        Tuple[float, float]: a tuple of (average loss, AUROC)
    """
    pipeline._model.eval()
    device = pipeline._device

    iterator = itertools.islice(iter(eval_dataloader), limit_batches)

    # We are using the AUROC for binary classification
    auroc = metrics.AUROC(task="binary").to(device)

    is_rank_zero = dist.get_rank() == 0
    if is_rank_zero:
        pbar = tqdm(
            iter(int, 1),
            desc=f"Evaluating {stage} set",
            total=len(eval_dataloader),
            disable=False,
        )
    
    total_loss = torch.tensor(0.0).to(device)  # Initialize total_loss as a tensor on the same device as _loss
    total_samples = 0
    with torch.no_grad():
        while True:
            try:
                _loss, logits, labels = pipeline.progress(map(transform_partial, iterator))
                # Calculating AUROC
                preds = torch.sigmoid(logits)
                auroc(preds, labels)
                # Calculating loss
                total_loss += _loss.detach()  # Detach _loss to prevent gradients from being calculated
                total_samples += len(labels)
                if is_rank_zero:
                    pbar.update(1)
            except StopIteration:
                break
    
    auroc_result = auroc.compute().item()
    average_loss = total_loss / total_samples if total_samples > 0 else torch.tensor(0.0).to(device)
    average_loss_value = average_loss.item()

    if is_rank_zero:
        print(f"Average loss over {stage} set: {average_loss_value:.4f}.")
        print(f"AUROC over {stage} set: {auroc_result}")
    
    return average_loss_value, auroc_result

# COMMAND ----------

def train(
    pipeline: TrainPipelineSparseDist,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    epoch: int,
    print_lr: bool,
    validation_freq: Optional[int],
    limit_train_batches: Optional[int],
    limit_val_batches: Optional[int]) -> None:
    """
    Trains model for 1 epoch. Helper function for train_val_test.

    Args:
        pipeline (TrainPipelineSparseDist): data pipeline.
        train_dataloader (DataLoader): Training set's dataloader.
        val_dataloader (DataLoader): Validation set's dataloader.
        epoch (int): The number of complete passes through the training set so far.
        print_lr (bool): Whether to print the learning rate every training step.
        validation_freq (Optional[int]): The number of training steps between validation runs within an epoch.
        limit_train_batches (Optional[int]): Limits the training set to the first `limit_train_batches` batches.
        limit_val_batches (Optional[int]): Limits the validation set to the first `limit_val_batches` batches.

    Returns:
        None.
    """
    pipeline._model.train()

    # Get the first `limit_train_batches` batches
    iterator = itertools.islice(iter(train_dataloader), limit_train_batches)

    # Only print out the progress bar on rank 0
    is_rank_zero = dist.get_rank() == 0
    if is_rank_zero:
        pbar = tqdm(
            iter(int, 1),
            desc=f"Epoch {epoch}",
            total=len(train_dataloader),
            disable=False,
        )

    # TorchRec's pipeline paradigm is unique as it takes in an iterator of batches for training.
    start_it = 0
    n = validation_freq if validation_freq else len(train_dataloader)
    for batched_iterator in batched(iterator, n):
        for it in itertools.count(start_it):
            try:
                if is_rank_zero and print_lr:
                    for i, g in enumerate(pipeline._optimizer.param_groups):
                        print(f"lr: {it} {i} {g['lr']:.6f}")
                pipeline.progress(map(transform_partial, batched_iterator))
                if is_rank_zero:
                    pbar.update(1)
            except StopIteration:
                if is_rank_zero:
                    print("Total number of iterations:", it)
                start_it = it
                break

        # If you are validating frequently, use the evaluation function
        if validation_freq and start_it % validation_freq == 0:
            evaluate(limit_val_batches, pipeline, val_dataloader, "val")
            pipeline._model.train()

# COMMAND ----------

def train_val_test(args, model, optimizer, device, train_dataloader, val_dataloader, test_dataloader) -> None:
    """
    Train/validation/test loop.

    Args:
        args (Args): args for training.
        model (torch.nn.Module): model to train.
        optimizer (torch.optim.Optimizer): optimizer to use.
        device (torch.device): device to use.
        train_dataloader (DataLoader): Training set's dataloader.
        val_dataloader (DataLoader): Validation set's dataloader.
        test_dataloader (DataLoader): Test set's dataloader.

    Returns:
        TrainValTestResults.
    """
    pipeline = TrainPipelineSparseDist(model, optimizer, device)
    
    # Getting base auroc and saving it to mlflow
    val_loss, val_auroc = evaluate(args.limit_val_batches, pipeline, val_dataloader, "val")
    if int(os.environ["RANK"]) == 0:
        mlflow.log_metric('val_loss', val_loss)
        mlflow.log_metric('val_auroc', val_auroc)

    # Running a training loop
    for epoch in range(args.epochs):
        train(
            pipeline,
            train_dataloader,
            val_dataloader,
            epoch,
            args.print_lr,
            args.validation_freq,
            args.limit_train_batches,
            args.limit_val_batches,
        )

        # Evaluate after each training epoch
        val_loss, val_auroc = evaluate(args.limit_val_batches, pipeline, val_dataloader, "val")
        if int(os.environ["RANK"]) == 0:
            mlflow.log_metric('val_loss', val_loss, step=epoch)
            mlflow.log_metric('val_auroc', val_auroc, step=epoch)

        # Save the underlying model and results to mlflow
        log_state_dict_to_mlflow(pipeline._model.module, artifact_path=f"model_state_dict_{epoch}")
    
    # Evaluate on the test set after training loop finishes
    test_loss, test_auroc = evaluate(args.limit_test_batches, pipeline, test_dataloader, "test")
    if int(os.environ["RANK"]) == 0:
        mlflow.log_metric('test_loss', test_loss)
        mlflow.log_metric('test_auroc', test_auroc)
    return test_auroc

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4. The Main Function
# MAGIC
# MAGIC This function trains the Two Tower recommendation model. For more information, see the following guides/docs/code:
# MAGIC
# MAGIC - https://pytorch.org/torchrec/
# MAGIC - https://github.com/pytorch/torchrec
# MAGIC - https://github.com/pytorch/torchrec/blob/main/examples/retrieval/two_tower_train.py#L75

# COMMAND ----------

# Update: Using the volume path defined previously. The definitions of the directories are
#         repeated here for convenience in case in subsequent runs you are skipping the above
#         section of the notebook.

# TODO: Specify where the data is stored in UC Volumes
# output_dir_train = "/Volumes/ml/recommender_systems/learning_from_sets_data/mds_train"
# output_dir_validation = "/Volumes/ml/recommender_systems/learning_from_sets_data/mds_validation"
# output_dir_test = "/Volumes/ml/recommender_systems/learning_from_sets_data/mds_test"

output_dir_train = os.path.join(mds_volume_path, "mds_train")
output_dir_validation = os.path.join(mds_volume_path, "mds_validation")
output_dir_test = os.path.join(mds_volume_path, "mds_test")

# Update: Define the local directories we need for Dataset efficiency as mentioned earlier.
#         Note that we reuse the directories here across runs, and that can sometimes cause
#         problems. It may be worth just skipping any potential performance benefit from
#         reusing them and setting them to some temporary directory instead.
local_dir_train = "/local_disk0/tmp/mds_train"
local_dir_validation = "/local_disk0/tmp/mds_validation"
local_dir_test = "/local_disk0/tmp/mds_test"

from torchrec.distributed.comm import get_local_size
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
from torchrec.distributed.model_parallel import (
    DistributedModelParallel,
    get_default_sharders,
)

def main(args: Args):
    import torch
    import mlflow
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    # Some preliminary torch setup
    torch.jit._state.disable()
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    backend = "nccl"
    torch.cuda.set_device(device)

    # Start MLflow
    os.environ['DATABRICKS_HOST'] = db_host
    os.environ['DATABRICKS_TOKEN'] = db_token

    # Save parameters to MLflow
    if global_rank == 0:
        # Update: Start a run explicitly, so we can have the run id available to 
        #         return later. Because it is in a distributed setting and we only
        #         want the MLflow parts to execute on world rank 0, it becomes a 
        #         little difficult to use the usual context manager. However, since
        #         in that context it is running on a different process, even if it 
        #         fails the run should still end when the process exits (I think).
        #         Also moved the set_experiment call into the rank guard since it 
        #         should not be relevant outside of that.
        experiment = mlflow.set_experiment(experiment_path)
        run = mlflow.start_run()
        run_id = run.info.run_id
        param_dict = get_relevant_fields(args, cat_cols, emb_counts)
        mlflow.log_params(param_dict)

    # Start distributed process group
    dist.init_process_group(backend=backend)

    # Loading the data
    # train_dataloader = get_dataloader_with_mosaic(output_dir_train, args.batch_size, "train")
    # val_dataloader = get_dataloader_with_mosaic(output_dir_validation, args.batch_size, "val")
    # test_dataloader = get_dataloader_with_mosaic(output_dir_test, args.batch_size, "test")
    train_dataloader = get_dataloader_with_mosaic(output_dir_train, local_dir_train, args.batch_size)
    val_dataloader = get_dataloader_with_mosaic(output_dir_validation, local_dir_validation, args.batch_size)
    test_dataloader = get_dataloader_with_mosaic(output_dir_test, local_dir_test, args.batch_size)

    # Create the embedding tables
    eb_configs = [
        EmbeddingBagConfig(
            name=f"t_{feature_name}",
            embedding_dim=args.embedding_dim,
            num_embeddings=emb_counts[feature_idx],
            feature_names=[feature_name],
        )
        for feature_idx, feature_name in enumerate(cat_cols)
    ]

    # Create the Two Tower model
    embedding_bag_collection = EmbeddingBagCollection(
        tables=eb_configs,
        device=torch.device("meta"),
    )
    two_tower_model = TwoTower(
        embedding_bag_collection=embedding_bag_collection,
        layer_sizes=args.layer_sizes,
        device=device,
    )
    two_tower_train_task = TwoTowerTrainTask(two_tower_model)
    apply_optimizer_in_backward(
        RowWiseAdagrad,
        two_tower_train_task.two_tower.ebc.parameters(),
        {"lr": args.learning_rate},
    )

    # Create a plan to shard the embedding tables across the GPUs and creating a distributed model
    planner = EmbeddingShardingPlanner(
        topology=Topology(
            local_world_size=get_local_size(),
            world_size=dist.get_world_size(),
            compute_device=device.type,
        ),
        batch_size=args.batch_size,
        # If you get an out-of-memory error, increase the percentage. See
        # https://pytorch.org/torchrec/torchrec.distributed.planner.html#torchrec.distributed.planner.storage_reservations.HeuristicalStorageReservation
        storage_reservation=HeuristicalStorageReservation(percentage=0.05),
    )
    plan = planner.collective_plan(
        two_tower_model, get_default_sharders(), dist.GroupMember.WORLD
    )
    model = DistributedModelParallel(
        module=two_tower_train_task,
        device=device,
    )

    # Print out the sharding information to see how the embedding tables are sharded across the GPUs
    if global_rank == 0 and args.print_sharding_plan:
        for collectionkey, plans in model._plan.plan.items():
            print(collectionkey)
            for table_name, plan in plans.items():
                print(table_name, "\n", plan, "\n")
    
    log_state_dict_to_mlflow(model.module.two_tower, artifact_path="model_state_dict_base")

    optimizer = KeyedOptimizerWrapper(
        dict(model.named_parameters()),
        lambda params: torch.optim.Adam(params, lr=args.learning_rate),
    )

    # Start the training loop
    results = train_val_test(
        args,
        model,
        optimizer,
        device,
        train_dataloader,
        val_dataloader,
        test_dataloader,
    )

    # Destroy the process group
    dist.destroy_process_group()

    # Update: If we are world rank zero and have a run id, return that as the return value
    #         from the distributed call, so we don't have to go manually look it up later.
    if global_rank == 0:
        mlflow.end_run()
        return run_id

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.5. Setting up MLflow
# MAGIC
# MAGIC **Note:** You must update the route for `db_host` to the URL of your Databricks workspace.

# COMMAND ----------

username = spark.sql("SELECT current_user()").first()['current_user()']
username

experiment_path = f'/Users/{username}/torchrec-learning-from-sets-example'
 
# TODO: Update the `db_host` with the URL for your Databricks workspace
db_context = dbutils.entry_point.getDbutils().notebook().getContext()
db_browser_hostname = db_context.tags().get("browserHostName").get()
db_host = f"https://{db_browser_hostname}"
db_token = db_context.apiToken().get()
 
# Manually create the experiment so that you know the id and can send that to the worker nodes when you scale later.
experiment = mlflow.set_experiment(experiment_path)

# COMMAND ----------

# MAGIC %md ## 4. Single Node + Single GPU Training
# MAGIC
# MAGIC Here, you set the environment variables to run training over the sample set of 100,000 data points (stored in Volumes in Unity Catalog and collected using Mosaic StreamingDataset). You can expect each epoch to take ~16 minutes.

# COMMAND ----------

# Update: I've been mostly running this on single node multi-GPU, though this
#         part may still work.

# os.environ["RANK"] = "0"
# os.environ["LOCAL_RANK"] = "0"
# os.environ["WORLD_SIZE"] = "1"
# os.environ["MASTER_ADDR"] = "localhost"
# os.environ["MASTER_PORT"] = "29500"

# args = Args()
# main(args)

# COMMAND ----------

# MAGIC %md ## 5. Single Node - Multi GPU Training
# MAGIC
# MAGIC This notebook uses TorchDistributor to handle training on a `g4dn.12xlarge` instance with 4 T4 GPUs. You can view the sharding plan in the output logs to see what tables are located on what GPUs. This takes ~8 minutes to run per epoch.
# MAGIC
# MAGIC **Note**: There may be cases where you receive unexpected errors (like the Python Kernel crashing or segmentation faults). This is a transient error and the easiest way to overcome it is to skip the single node single GPU training code before you run any distributed code (single node multi GPU or multi node multi GPU).
# MAGIC
# MAGIC **Note**: If you see any errors that are associated with Mosaic Data Loading, these are transient errors that can be overcome by rerunning the failed cell.

# COMMAND ----------

args = Args()
# Update: The learning rate in the default seems way to high judging from initial runs. 
#         Adjusting it lower helps a lot. The model seems to converge after around 
#         5 to 10 epochs. We could consider adding early stopping here and so forth.
args.learning_rate = 0.0001
args.epochs = 10
run_id = TorchDistributor(num_processes=4, local_mode=True, use_gpu=True).run(main, args)
assert run_id is not None, "TorchDistributor did not return the run_id"
print(run_id)

# COMMAND ----------

# MAGIC %md ## 6. Multi Node + Multi GPU Training
# MAGIC
# MAGIC This is tested with a `g4dn.12xlarge` instance as a worker (with 4 T4 GPUs). You can view the sharding plan in the output logs to see what tables are located on what GPUs. This takes ~6 minutes to run per epoch.
# MAGIC
# MAGIC **Note**: There may be cases where you receive unexpected errors (like the Python Kernel crashing or segmentation faults). This is a transient error and the easiest way to overcome it is to skip the single node single GPU training code before you run any distributed code (single node multi GPU or multi node multi GPU).
# MAGIC
# MAGIC **Note**: If you see any errors that are associated with Mosaic Data Loading, these are transient errors that can be overcome by rerunning the failed cell.

# COMMAND ----------

# args = Args()
# TorchDistributor(num_processes=4, local_mode=False, use_gpu=True).run(main, args)

# COMMAND ----------

# MAGIC %md ## 7. Inference
# MAGIC
# MAGIC Because the Two Tower Model's `state_dict`s are logged to MLflow, you can use the following code to load any of the saved `state_dict`s and create the associated Two Tower model with it. You can further expand this by 1) saving the loaded model to mlflow for inference or 2) doing batch inference using a UDF.
# MAGIC
# MAGIC Note: The saving code and loading code is used for loading the entire Two Tower model on one node and is useful as an example. In real world use cases, the expected model size could be significant (as the embedding tables can scale with the number of users or the number of products and items). It might be worthwhile to consider distributed inference.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.1. Creating the Two Tower model from saved `state_dict`
# MAGIC
# MAGIC **Note:** You must update this with the correct `run_id` and path to the MLflow artifact.

# COMMAND ----------

def get_mlflow_model(run_id, artifact_path="model_state_dict"):
    from mlflow import MlflowClient

    device = torch.device("cuda")
    run = mlflow.get_run(run_id)
    
    cat_cols = eval(run.data.params.get('cat_cols'))
    emb_counts = eval(run.data.params.get('emb_counts'))
    layer_sizes = eval(run.data.params.get('layer_sizes'))
    embedding_dim = eval(run.data.params.get('embedding_dim'))

    MlflowClient().download_artifacts(run_id, f"{artifact_path}/state_dict.pth", "/databricks/driver")
    state_dict = mlflow.pytorch.load_state_dict(f"/databricks/driver/{artifact_path}")
    
    # Remove the prefix "two_tower." from all of the keys in the state_dict
    state_dict = {k[10:]: v for k, v in state_dict.items()}

    eb_configs = [
        EmbeddingBagConfig(
            name=f"t_{feature_name}",
            embedding_dim=embedding_dim,
            num_embeddings=emb_counts[feature_idx],
            feature_names=[feature_name],
        )
        for feature_idx, feature_name in enumerate(cat_cols)
    ]

    embedding_bag_collection = EmbeddingBagCollection(
        tables=eb_configs,
        device=device,
    )
    two_tower_model = TwoTower(
        embedding_bag_collection=embedding_bag_collection,
        layer_sizes=layer_sizes,
        device=device,
    )

    two_tower_model.load_state_dict(state_dict)

    return two_tower_model, cat_cols, emb_counts

# Load the model (epoch 2) from the MLflow run
# TODO: Update this with the correct run_id and path
# two_tower_model, cat_cols, emb_counts = get_mlflow_model("1307c11d143347dea64a691a308f19ed", artifact_path="model_state_dict_1")
two_tower_model, cat_cols, emb_counts = get_mlflow_model(run_id, artifact_path="model_state_dict_1")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.2. Helper Function to Transform Dataloader to Two Tower Inputs
# MAGIC
# MAGIC The inputs that Two Tower expects are: `sparse_features`, so this section reuses aspects of the code from Section 3.4.2. The code shown here is verbose for clarity.

# COMMAND ----------

def transform_test(batch, cat_cols, emb_counts):
    kjt_values: List[int] = []
    kjt_lengths: List[int] = []
    for col_idx, col_name in enumerate(cat_cols):
        values = batch[col_name]
        for value in values:
            if value:
                kjt_values.append(
                    value % emb_counts[col_idx]
                )
                kjt_lengths.append(1)
            else:
                kjt_lengths.append(0)

    sparse_features = KeyedJaggedTensor.from_lengths_sync(
        cat_cols,
        torch.tensor(kjt_values),
        torch.tensor(kjt_lengths, dtype=torch.int32),
    )
    return sparse_features

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.3. Getting the Data

# COMMAND ----------

num_batches = 5 # Number of batches to print out at a time 
batch_size = 1 # Print out each individual row

# TODO: Update this path to point to the test dataset stored in UC Volumes
# test_data_path = "/Volumes/ml/recommender_systems/learning_from_sets_data/mds_test"
# test_data_path = "/Volumes/users/corey_abshire/learning_from_sets_data/mds_test"
# test_dataloader = iter(get_dataloader_with_mosaic(test_data_path, batch_size, "test"))

test_dataloader = iter(get_dataloader_with_mosaic(output_dir_test, local_dir_test + "_2", batch_size))

# COMMAND ----------

# MAGIC %md ### 7.4. Running Tests
# MAGIC
# MAGIC In this example, you ran training for 3 epochs. The results were reasonable. Running a larger number of epochs would likely lead to optimal performance.

# COMMAND ----------

for _ in range(num_batches):
    device = torch.device("cuda:0")
    two_tower_model.to(device)
    two_tower_model.eval()

    next_batch = next(test_dataloader)
    expected_result = next_batch["label"][0]
    
    sparse_features = transform_test(next_batch, cat_cols, emb_counts)
    sparse_features = sparse_features.to(device)
    
    query_embedding, candidate_embedding = two_tower_model(kjt=sparse_features)
    actual_result = (query_embedding * candidate_embedding).sum(dim=1).squeeze()
    actual_result = torch.sigmoid(actual_result)
    print(f"Expected Result: {expected_result}; Actual Result: {actual_result.round().item()}")

# COMMAND ----------

# MAGIC %md ## 8. Model Serving and Vector Search
# MAGIC
# MAGIC For information about how to serve the model, see the Databricks Model Serving documentation ([AWS](https://docs.databricks.com/en/machine-learning/model-serving/index.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/)).
# MAGIC
# MAGIC Also, the Two Tower model is unique as it generates a `query` and `candidate` embedding, and therefore, allows you to create a vector index of movies, and then allows you to find the K movies that a user (given their generated vector) would most likely give a high rating. For more information, view the code [here](https://github.com/pytorch/torchrec/blob/main/examples/retrieval/two_tower_train.py#L198) for how to create your own FAISS Index. You can also take a similar approach with Databricks Vector Search ([AWS](https://docs.databricks.com/en/generative-ai/vector-search.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/vector-search/)).
