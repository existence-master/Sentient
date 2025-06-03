# Usage:
# 1. Authenticate with Google: (Run this from the parent directory of client)
# python -m client.app.client --user-id myuser auth-google
#
# 2. Execute an action item: (Run this from the parent directory of client)
# python client.py --user-id myuser execute "Prepare a presentation for the Quarterly Report meeting."

import click
from google_auth_oauthlib.flow import InstalledAppFlow
import requests
import json

SERVER_URL = "http://localhost:8000"

@click.group()
@click.option('--user-id', required=True, help="Unique identifier for the user")
@click.pass_context
def cli(ctx, user_id):
    ctx.ensure_object(dict)
    ctx.obj['user_id'] = user_id

@cli.command()
def auth_google():
    """Authenticate with Google and send token to the server."""
    user_id = click.get_current_context().obj['user_id']
    scopes = [
        'https://www.googleapis.com/auth/gmail.readonly',
        'https://www.googleapis.com/auth/gmail.send',
        'https://www.googleapis.com/auth/calendar',
        'https://www.googleapis.com/auth/drive',
        'https://www.googleapis.com/auth/documents',
        'https://www.googleapis.com/auth/presentations',
        'https://www.googleapis.com/auth/spreadsheets'
    ]
    flow = InstalledAppFlow.from_client_secrets_file(
        'client/client_secrets.json',  # Must be provided by the user
        scopes=scopes
    )
    creds = flow.run_local_server(port=0)
    token_data = {
        'refresh_token': creds.refresh_token,
        'token': creds.token,
        'expires_at': creds.expiry.isoformat(),
        'client_id': creds.client_id,
        'client_secret': creds.client_secret,
        'scopes': creds.scopes
    }
    response = requests.post(f"{SERVER_URL}/auth/google", json={'user_id': user_id, 'token_data': token_data})
    click.echo(response.json())

@cli.command()
@click.argument('action_item')
def execute(action_item):
    """Send an action item to the server for execution."""
    user_id = click.get_current_context().obj['user_id']
    response = requests.post(f"{SERVER_URL}/execute", json={'user_id': user_id, 'action_item': action_item})
    click.echo(response.json())

if __name__ == '__main__':
    cli()