def create_presentation(title):
    service = get_slides_service()

    # プレゼンテーションの作成
    presentation = service.presentations().create(body={"title": title}).execute()
    presentation_id = presentation["presentationId"]
    
    print(f"✅ Successfully created a presentation '{title}'.")
    print(f"https://docs.google.com/presentation/d/{presentation_id}/edit")
    return presentation_id

def insert_image(presentation_id, slide_index, image_url, x, y, width, height):
    service = get_slides_service()
    
    # スライド情報の取得
    presentation = service.presentations().get(presentationId=presentation_id).execute()
    slide_id = presentation["slides"][slide_index]["objectId"]
    
    # 画像挿入リクエスト
    requests = [
        {
            "createImage": {
                "url": image_url,
                "elementProperties": {
                    "pageObjectId": slide_id,
                    "size": {
                        "height": {"magnitude": height, "unit": "PT"},
                        "width": {"magnitude": width, "unit": "PT"}
                    },
                    "transform": {
                        "scaleX": 1,
                        "scaleY": 1,
                        "translateX": x,
                        "translateY": y,
                        "unit": "PT"
                    }
                }
            }
        }
    ]
    
    # リクエストの実行
    service.presentations().batchUpdate(presentationId=presentation_id, body={"requests": requests}).execute()
    print(f"✅ Inserted image '{image_url}' to slide No. {slide_index+1}.")


def insert_image(presentation_id, slide_index, image_url, x, y, width, height):
    service = get_slides_service()
    
    # スライド情報の取得
    presentation = service.presentations().get(presentationId=presentation_id).execute()
    slide_id = presentation["slides"][slide_index]["objectId"]
    
    # 画像挿入リクエスト
    requests = [
        {
            "createImage": {
                "url": image_url,
                "elementProperties": {
                    "pageObjectId": slide_id,
                    "size": {
                        "height": {"magnitude": height, "unit": "PT"},
                        "width": {"magnitude": width, "unit": "PT"}
                    },
                    "transform": {
                        "scaleX": 1,
                        "scaleY": 1,
                        "translateX": x,
                        "translateY": y,
                        "unit": "PT"
                    }
                }
            }
        }
    ]
    
    # リクエストの実行
    service.presentations().batchUpdate(presentationId=presentation_id, body={"requests": requests}).execute()
    print(f"✅ 画像 '{image_url}' をスライド {slide_index+1} に挿入しました。")


import pandas as pd

def insert_text_from_csv(presentation_id, slide_index, csv_path, x, y, font_size=24):
    service = get_slides_service()
    
    # CSV データの読み込み
    df = pd.read_csv(csv_path)
    
    # スライド情報の取得
    presentation = service.presentations().get(presentationId=presentation_id).execute()
    slide_id = presentation["slides"][slide_index]["objectId"]
    
    # CSV データを特定のセルから取得
    text_value = str(df.iloc[0, 0])  # 1行1列目のデータを取得
    
    # テキスト挿入リクエスト
    requests = [
        {
            "createShape": {
                "shapeType": "TEXT_BOX",
                "elementProperties": {
                    "pageObjectId": slide_id,
                    "size": {
                        "height": {"magnitude": 50, "unit": "PT"},
                        "width": {"magnitude": 300, "unit": "PT"}
                    },
                    "transform": {
                        "scaleX": 1,
                        "scaleY": 1,
                        "translateX": x,
                        "translateY": y,
                        "unit": "PT"
                    }
                },
                "text": {
                    "textElements": [
                        {
                            "textRun": {
                                "content": text_value,
                                "style": {
                                    "fontSize": {"magnitude": font_size, "unit": "PT"}
                                }
                            }
                        }
                    ]
                }
            }
        }
    ]
    
    # リクエストの実行
    service.presentations().batchUpdate(presentationId=presentation_id, body={"requests": requests}).execute()
    print(f"✅ Inserted csv's contents '{text_value}' to slide no. {slide_index+1}.")

def add_slide(presentation_id, layout="TITLE_AND_CONTENT"):
    service = get_slides_service()

    # プレゼンテーション情報の取得
    presentation = service.presentations().get(presentationId=presentation_id).execute()
    num_slides = len(presentation["slides"])
    
    # スライド挿入リクエスト
    requests = [
        {
            "createSlide": {
                "insertionIndex": num_slides,
                "slideLayoutReference": {"predefinedLayout": layout}
            }
        }
    ]
    
    # リクエストの実行
    service.presentations().batchUpdate(presentationId=presentation_id, body={"requests": requests}).execute()
    print(f"✅ Added a new page (layout: {layout}).")

