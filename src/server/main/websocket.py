import logging 
from typing import Dict, Any, List
from fastapi import WebSocket, status, WebSocketDisconnect
from starlette.websockets import WebSocketState 

logger = logging.getLogger(__name__) 

class MainWebSocketManager:
    def __init__(self):
        self.voice_connections: Dict[str, WebSocket] = {}
        self.notification_connections: Dict[str, WebSocket] = {}
        logger.info("MainWebSocketManager initialized.")

    async def connect_voice(self, websocket: WebSocket, user_id: str):
        if user_id in self.voice_connections:
            old_ws = self.voice_connections.pop(user_id, None)
            if old_ws and old_ws.client_state == WebSocketState.CONNECTED:
                try: 
                    await old_ws.close(code=status.WS_1000_NORMAL_CLOSURE, reason="New voice connection by same user.")
                    logger.info(f"Closed old voice WebSocket for user: {user_id}")
                except Exception as e: 
                    logger.error(f"Error closing old voice WS for {user_id}: {e}")
        
        self.voice_connections[user_id] = websocket
        logger.info(f"Voice WebSocket connected for user: {user_id}. Total voice connections: {len(self.voice_connections)}")

    async def disconnect_voice(self, websocket: WebSocket):
        uid_to_remove = None
        for uid, ws in self.voice_connections.items():
            if ws == websocket:
                uid_to_remove = uid
                break
        
        if uid_to_remove and uid_to_remove in self.voice_connections:
            del self.voice_connections[uid_to_remove]
            logger.info(f"Voice WebSocket disconnected for user: {uid_to_remove}. Total voice connections: {len(self.voice_connections)}")

    async def connect_notifications(self, websocket: WebSocket, user_id: str):
        if user_id in self.notification_connections:
            old_ws = self.notification_connections.pop(user_id, None)
            if old_ws and old_ws.client_state == WebSocketState.CONNECTED:
                try: 
                    await old_ws.close(code=status.WS_1000_NORMAL_CLOSURE, reason="New notification connection by same user.")
                    logger.info(f"Closed old notification WebSocket for user: {user_id}")
                except Exception as e: 
                    logger.error(f"Error closing old notif WS for {user_id}: {e}")

        self.notification_connections[user_id] = websocket
        logger.info(f"Notification WebSocket connected for user: {user_id}. Total notification connections: {len(self.notification_connections)}")

    async def disconnect_notifications(self, websocket: WebSocket):
        uid_to_remove = None
        for uid, ws in self.notification_connections.items():
            if ws == websocket:
                uid_to_remove = uid
                break
        
        if uid_to_remove and uid_to_remove in self.notification_connections:
            del self.notification_connections[uid_to_remove]
            logger.info(f"Notification WebSocket disconnected for user: {uid_to_remove}. Total notification connections: {len(self.notification_connections)}")

    async def send_personal_json_message(self, message_data: Dict[str, Any], user_id: str, connection_type: str = "notifications"):
        connections_dict = self.notification_connections if connection_type == "notifications" else self.voice_connections
        websocket = connections_dict.get(user_id)
        
        if websocket and websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_json(message_data)
            except (WebSocketDisconnect, RuntimeError) as e:
                logger.warning(f"Failed to send to {user_id} ({connection_type}), disconnecting. Error: {e}")
                await self.disconnect_by_type(websocket, connection_type)
        elif websocket:
            logger.warning(f"Attempted to send to user {user_id} ({connection_type}) but WebSocket state is {websocket.client_state}. Cleaning up.")
            await self.disconnect_by_type(websocket, connection_type)

    async def broadcast_json_to_all_notifications(self, message_data: Dict[str, Any]):
        sockets_to_remove: List[WebSocket] = []
        
        for user_id, websocket in list(self.notification_connections.items()): 
            if websocket.client_state == WebSocketState.CONNECTED:
                try:
                    await websocket.send_json(message_data)
                except (WebSocketDisconnect, RuntimeError) as e:
                    logger.warning(f"Failed to broadcast to {user_id}, disconnecting. Error: {e}")
                    sockets_to_remove.append(websocket)
            else:
                sockets_to_remove.append(websocket)

        for ws_to_remove in sockets_to_remove:
            await self.disconnect_notifications(ws_to_remove)

    async def disconnect_by_type(self, websocket: WebSocket, connection_type: str):
        if connection_type == "notifications":
            await self.disconnect_notifications(websocket)
        else:
            await self.disconnect_voice(websocket)