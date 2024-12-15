import os
import traceback
from datetime import datetime
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv
from minio import Minio
from minio.commonconfig import CopySource

load_dotenv(find_dotenv())

access_key = os.environ.get("MINIO_ACCESS_KEY")
secret_key = os.environ.get("MINIO_SECRET_KEY")

client = Minio(
    "data.pyrogn.ru",
    access_key=access_key,
    secret_key=secret_key,
    secure=True,
    # http_client=None,
    # cert_check=,
)

BUCKET_NAME = "tasksai"
KEEP_N_BACKUPS = 5


@click.group()
def cli():
    """Sync local ./data with MinIO bucket"""
    pass


@cli.command()
def pull():
    """Pull all data from MinIO to local ./data folder"""
    try:
        Path.mkdir(Path("data"), exist_ok=True, parents=True)

        objects = client.list_objects(BUCKET_NAME, recursive=True)
        for obj in objects:
            local_path = Path("data") / obj.object_name
            Path.mkdir(Path(local_path).parent, exist_ok=True, parents=True)
            client.fget_object(BUCKET_NAME, obj.object_name, local_path)
            click.echo(f"Downloaded: {obj.object_name}")

        click.echo("Pull completed successfully")

    except Exception as e:
        click.echo(f"Error: {e} {traceback.format_exc()}", err=True)


@cli.command()
def push():
    """Push local ./data folder to MinIO (with backup)"""
    try:
        if not client.bucket_exists(BUCKET_NAME):
            client.make_bucket(BUCKET_NAME)
            click.echo(f"Created bucket: {BUCKET_NAME}")

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        backup_bucket = f"{BUCKET_NAME}-backup-{timestamp}"

        # Backup existing data
        if client.bucket_exists(BUCKET_NAME):
            client.make_bucket(backup_bucket)
            objects = client.list_objects(BUCKET_NAME, recursive=True)
            for obj in objects:
                source = CopySource(BUCKET_NAME, obj.object_name)
                client.copy_object(backup_bucket, obj.object_name, source)
            click.echo(f"Backup created in bucket: {backup_bucket}")

        all_buckets = [
            bucket.name
            for bucket in client.list_buckets()
            if bucket.name.startswith(f"{BUCKET_NAME}-backup-")
        ]
        all_buckets.sort(reverse=True)

        for old_bucket in all_buckets[KEEP_N_BACKUPS:]:
            objects = client.list_objects(old_bucket, recursive=True)
            for obj in objects:
                client.remove_object(old_bucket, obj.object_name)
            client.remove_bucket(old_bucket)
            click.echo(f"Removed old backup bucket: {old_bucket}")

        for root, _, files in os.walk("data"):
            for file in files:
                local_path = Path(root) / file
                object_name = os.path.relpath(local_path, "data")
                client.fput_object(BUCKET_NAME, object_name, local_path)
                click.echo(f"Uploaded: {object_name}")

        click.echo("Push completed successfully")

    except Exception as e:
        click.echo(f"Error: {e} {traceback.format_exc()}", err=True)


if __name__ == "__main__":
    cli()
