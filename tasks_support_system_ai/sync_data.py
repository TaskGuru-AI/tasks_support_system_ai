import click
from minio import Minio
import os
from datetime import datetime
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

access_key = os.environ.get("MINIO_ACCESS_KEY")
secret_key = os.environ.get("MINIO_SECRET_KEY")

client = Minio(
    "data.pyrogn.ru:9000",
    access_key=access_key,
    secret_key=secret_key,
    secure=False,
    # http_client=None,
    cert_check=False,
)

BUCKET_NAME = "tasksai"


@click.group()
def cli():
    """Sync local ./data with MinIO bucket"""
    pass


@cli.command()
def pull():
    """Pull all data from MinIO to local ./data folder"""
    try:
        os.makedirs("data", exist_ok=True)

        objects = client.list_objects(BUCKET_NAME)
        for obj in objects:
            local_path = os.path.join("data", obj.object_name)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            client.fget_object(BUCKET_NAME, obj.object_name, local_path)
            click.echo(f"Downloaded: {obj.object_name}")

        click.echo("Pull completed successfully")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@cli.command()
def push():
    """Push local ./data folder to MinIO (with backup)"""
    try:
        if not client.bucket_exists(BUCKET_NAME):
            client.make_bucket(BUCKET_NAME)
            click.echo(f"Created bucket: {BUCKET_NAME}")

        # Create backup of existing data
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        backup_bucket = f"tasksai-backup-{timestamp}"

        if client.bucket_exists(BUCKET_NAME):
            client.make_bucket(backup_bucket)
            objects = client.list_objects(BUCKET_NAME)
            for obj in objects:
                client.copy_object(
                    backup_bucket, obj.object_name, f"{BUCKET_NAME}/{obj.object_name}"
                )
            click.echo(f"Backup created in bucket: {backup_bucket}")

        # Upload all files from ./data
        for root, _, files in os.walk("data"):
            for file in files:
                local_path = os.path.join(root, file)
                # Create object name relative to data directory
                object_name = os.path.relpath(local_path, "data")
                # Upload file
                client.fput_object(BUCKET_NAME, object_name, local_path)
                click.echo(f"Uploaded: {object_name}")

        click.echo("Push completed successfully")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)


if __name__ == "__main__":
    cli()
