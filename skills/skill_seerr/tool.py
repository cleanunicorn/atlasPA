import json
import requests
from typing import Optional, Dict, Any

SEERR_HOST = "http://192.168.25.1:5055"
SEERR_API_KEY = "86tbNkcfsitw7Q@V9tZr"

def run(input: str) -> Dict[str, Any]:
    """Execute seerr skill action"""
    parts = input.split("|")
    action = parts[0]
    args = {}
    
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            args[key] = value
    
    headers = {"X-API-Key": SEERR_API_KEY}
    
    if action == "check_status":
        try:
            resp = requests.get(f"{SEERR_HOST}/api/health", headers=headers, timeout=10)
            return {"status": "ok", "data": resp.json() if resp.status_code == 200 else resp.text}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    elif action == "list_pending":
        try:
            resp = requests.get(f"{SEERR_HOST}/api/requests", headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                movies = [m for m in data if m.get("type") == "movie"]
                return {"status": "ok", "count": len(movies), "pending": movies}
            return {"status": "error", "message": resp.text}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    elif action == "get_media_status":
        tmdb_id = args.get("tmdb_id", args.get("id"))
        if not tmdb_id:
            return {"status": "error", "message": "tmdb_id required"}
        try:
            resp = requests.get(f"{SEERR_HOST}/api/requests?tmdbId={tmdb_id}", headers=headers, timeout=10)
            if resp.status_code == 200:
                return {"status": "ok", "data": resp.json()}
            return {"status": "error", "message": resp.text}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    elif action == "add_media":
        tmdb_id = args.get("tmdb_id")
        if not tmdb_id:
            return {"status": "error", "message": "tmdb_id required for add_media"}
        try:
            resp = requests.post(
                f"{SEERR_HOST}/api/requests",
                headers=headers,
                json={"tmdbId": int(tmdb_id), "type": "movie"},
                timeout=10
            )
            if resp.status_code == 200:
                return {"status": "ok", "message": "Media request added successfully", "data": resp.json()}
            return {"status": "error", "message": resp.text}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    elif action in ["approve_media", "decline_media"]:
        request_id = args.get("id")
        if not request_id:
            return {"status": "error", "message": "id required"}
        try:
            endpoint = f"{SEERR_HOST}/api/requests/{request_id}/approve" if action == "approve_media" else f"{SEERR_HOST}/api/requests/{request_id}/decline"
            resp = requests.post(endpoint, headers=headers, timeout=10)
            if resp.status_code in [200, 201]:
                return {"status": "ok", "message": f"{action} successful"}
            return {"status": "error", "message": resp.text}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    elif action == "remove_media":
        request_id = args.get("id")
        if not request_id:
            return {"status": "error", "message": "id required"}
        try:
            resp = requests.delete(f"{SEERR_HOST}/api/requests/{request_id}", headers=headers, timeout=10)
            if resp.status_code in [200, 204]:
                return {"status": "ok", "message": "Media request removed"}
            return {"status": "error", "message": resp.text}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    elif action == "get_activity":
        limit = int(args.get("limit", 10))
        try:
            resp = requests.get(f"{SEERR_HOST}/api/activity?limit={limit}", headers=headers, timeout=10)
            if resp.status_code == 200:
                return {"status": "ok", "data": resp.json()}
            return {"status": "error", "message": resp.text}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    else:
        return {"status": "error", "message": f"Unknown action: {action}"}
