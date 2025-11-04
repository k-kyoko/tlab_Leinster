from google.oauth2 import service_account
from googleapiclient.discovery import build

# API 認証情報を読み込む
def get_slides_service():
    # credentials.json のパス
    creds_path = "credentials.json"
    
    # 認証情報を読み込む
    creds = service_account.Credentials.from_service_account_file(creds_path, scopes=["https://www.googleapis.com/auth/presentations"])
    
    # Google Slides API サービスを構築
    service = build("slides", "v1", credentials=creds)
    return service
