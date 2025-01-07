import os
import traceback
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv
from minio import Minio

load_dotenv(find_dotenv())

MINIO_DOMAIN = "data.pyrogn.ru"
BUCKET_NAME = "tasksai"
KEEP_N_BACKUPS = 10

access_key = os.environ.get("MINIO_ACCESS_KEY")
secret_key = os.environ.get("MINIO_SECRET_KEY")

client = Minio(
    MINIO_DOMAIN,
    access_key=access_key,
    secret_key=secret_key,
    secure=True,
)


@click.group()
def cli():
    """Sync local ./data with MinIO bucket"""
    pass


@cli.command()
def pull():
    """Pull all data from MinIO to local ./data folder"""
    try:
        data_path = Path("data")
        data_path.mkdir(exist_ok=True, parents=True)

        objects = client.list_objects(BUCKET_NAME, recursive=True)
        for obj in objects:
            local_path = data_path / obj.object_name
            local_path.parent.mkdir(exist_ok=True, parents=True)
            client.fget_object(BUCKET_NAME, obj.object_name, local_path)
            click.echo(f"Downloaded: {obj.object_name}")

        if os.name != "nt":
            for path in data_path.rglob("*"):
                try:
                    path.chmod(0o777 if path.is_dir() else 0o666)
                except Exception as e:
                    click.echo(f"Warning: Could not set permissions for {path}: {e}", err=True)

        click.echo("Pull completed successfully")

    except Exception as e:
        click.echo(f"Error: {e} {traceback.format_exc()}", err=True)


@cli.command()
def push():
    """Push local ./data folder to MinIO with versioning"""
    try:
        # Create bucket if it doesn't exist and enable versioning
        if not client.bucket_exists(BUCKET_NAME):
            client.make_bucket(BUCKET_NAME)
            print(client.get_bucket_versioning(BUCKET_NAME))
            client.set_bucket_versioning(BUCKET_NAME, True)
            click.echo(f"Created versioned bucket: {BUCKET_NAME}")

        # Upload files
        for file_path in Path("data").rglob("*"):
            if file_path.is_file():
                object_name = str(file_path.relative_to("data").as_posix())
                client.fput_object(BUCKET_NAME, object_name, file_path)
                click.echo(f"Uploaded: {object_name}")

        click.echo("Push completed successfully")

    except Exception as e:
        click.echo(f"Error: {e} {traceback.format_exc()}", err=True)


if __name__ == "__main__":
    cli()
