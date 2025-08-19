# Outlook Integration for Sentient

This module provides Outlook email integration for the Sentient AI assistant using Microsoft Graph API.

## Features

- **Read Emails**: List and read emails from different folders (Inbox, Sent Items, etc.)
- **Send Emails**: Compose and send new emails
- **Reply to Emails**: Reply to existing email threads
- **Search Emails**: Search for specific emails using Microsoft Graph search
- **Manage Folders**: List and navigate email folders
- **Privacy Filters**: Apply user-defined privacy filters to email content

## Setup

### 1. Microsoft Azure App Registration

1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to "Azure Active Directory" > "App registrations"
3. Click "New registration"
4. Fill in the details:
   - **Name**: Sentient Outlook Integration
   - **Supported account types**: Accounts in any organizational directory and personal Microsoft accounts
   - **Redirect URI**: Web - `https://your-domain.com/integrations/oauth/callback`

### 2. Configure API Permissions

1. In your app registration, go to "API permissions"
2. Click "Add a permission"
3. Select "Microsoft Graph"
4. Choose "Delegated permissions"
5. Add the following permissions:
   - `Mail.Read` - Read user mail
   - `Mail.Send` - Send mail as a user
   - `User.Read` - Sign in and read user profile

### 3. Environment Variables

Add the following environment variables to your `.env` file:

```bash
# Outlook OAuth Configuration
OUTLOOK_CLIENT_ID=your_azure_app_client_id
OUTLOOK_CLIENT_SECRET=your_azure_app_client_secret

# Outlook MCP Server URL (optional, defaults to localhost:9027)
OUTLOOK_MCP_SERVER_URL=http://localhost:9027/sse
```

### 4. Start the Outlook MCP Server

```bash
cd src/server/mcp_hub/outlook
python main.py
```

The server will start on port 9027 by default.

## Usage

### Available Tools

1. **get_emails**: Retrieve emails from a specific folder
   - Parameters: `folder`, `top`, `skip`, `search`

2. **get_email**: Get a specific email by ID
   - Parameters: `message_id`

3. **send_email**: Send a new email
   - Parameters: `subject`, `body`, `to_recipients`, `cc_recipients`, `bcc_recipients`

4. **reply_to_email**: Reply to an existing email
   - Parameters: `message_id`, `body`, `cc_recipients`, `bcc_recipients`

5. **get_folders**: List email folders
   - Parameters: None

6. **search_emails**: Search for emails
   - Parameters: `query`, `top`

### Example Usage

```python
# Get recent emails from inbox
emails = await get_emails(folder="inbox", top=10)

# Send an email
result = await send_email(
    subject="Test Email",
    body="<p>This is a test email.</p>",
    to_recipients=["recipient@example.com"]
)

# Search for emails
search_results = await search_emails(query="meeting", top=5)
```

## Privacy and Security

- All credentials are encrypted using AES encryption
- User privacy filters are applied to email content
- Access tokens are stored securely in MongoDB
- The integration respects Microsoft's data handling policies

## Troubleshooting

### Common Issues

1. **OAuth Error**: Ensure your redirect URI matches exactly in Azure app registration
2. **Permission Denied**: Verify all required API permissions are granted
3. **Token Expired**: The integration handles token refresh automatically
4. **Connection Issues**: Check that the MCP server is running on the correct port

### Debug Mode

Enable debug logging by setting the environment variable:
```bash
ENVIRONMENT=dev-local
```

## API Reference

The integration uses Microsoft Graph API v1.0. For detailed API documentation, visit:
https://docs.microsoft.com/en-us/graph/api/overview

## Contributing

When contributing to this integration:

1. Follow the existing code patterns
2. Add appropriate error handling
3. Include privacy filter considerations
4. Update this README with any new features
5. Test thoroughly with different email scenarios
