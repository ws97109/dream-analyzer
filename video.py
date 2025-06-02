import gradio as gr
import tempfile
import os
from PIL import Image
import io
import time

# 本地模型圖片轉影片（示意）
def local_image_to_video(image_path, motion_strength, randomness, seed):
    # 模擬處理：這裡應該是你本地模型的實作
    from moviepy.editor import ImageClip
    clip = ImageClip(image_path).set_duration(2).resize(height=512)
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    clip.write_videofile(temp_video.name, fps=10, codec="libx264")
    return temp_video.name

def generate_video(input_image, motion_strength, randomness, seed_value, progress=gr.Progress()):
    """
    Gradio 介面的影片生成函數 (本地模型)
    """
    if input_image is None:
        return None, "❌ 請上傳一張圖片！"
    try:
        progress(0.1, desc="準備處理圖片...")
        # 保存上傳的圖片到臨時文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            if hasattr(input_image, 'save'):
                input_image.save(tmp_file.name, 'JPEG')
            else:
                with open(input_image, 'rb') as f:
                    tmp_file.write(f.read())
            tmp_image_path = tmp_file.name
        progress(0.3, desc="生成影片中...")
        video_path = local_image_to_video(
            tmp_image_path,
            motion_strength,
            randomness,
            seed_value
        )
        progress(0.8, desc="處理生成結果...")
        os.unlink(tmp_image_path)
        progress(1.0, desc="完成！")
        return video_path, "✅ 影片生成成功！"
    except Exception as e:
        return None, f"❌ 生成失敗: {str(e)}"

def create_gradio_interface():
    """創建 Gradio 介面"""
    custom_css = """
    .main-container { max-width: 1200px; margin: 0 auto; }
    .header { text-align: center; margin-bottom: 30px; }
    .status-box { padding: 10px; border-radius: 5px; margin: 10px 0; }
    .footer { text-align: center; margin-top: 30px; color: #666; }
    """
    with gr.Blocks(css=custom_css, title="圖片生成影片 - AI 工具") as demo:
        gr.Markdown("""
        # 🎬 圖片生成影片 AI 工具
        使用本地模型將靜態圖片轉換為動態影片
        """, elem_classes="header")
        with gr.Row():
            with gr.Column(scale=2):
                # 圖片上傳區域
                gr.Markdown("### 📷 上傳圖片")
                input_image = gr.Image(
                    label="選擇要轉換的圖片",
                    type="pil",
                    height=300
                )
                # 參數設置
                gr.Markdown("### ⚙️ 生成參數")
                with gr.Group():
                    motion_strength = gr.Slider(
                        minimum=1,
                        maximum=255,
                        value=127,
                        step=1,
                        label="動作強度",
                        info="數值越高，影片中的動作越明顯"
                    )
                    randomness = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.02,
                        step=0.01,
                        label="隨機性",
                        info="控制生成的多樣性"
                    )
                    seed_value = gr.Number(
                        label="隨機種子",
                        value=42,
                        precision=0,
                        info="相同種子會產生相似結果"
                    )
            with gr.Column(scale=2):
                # 生成按鈕
                gr.Markdown("### 🚀 生成影片")
                generate_button = gr.Button(
                    "開始生成影片",
                    variant="primary",
                    size="lg"
                )
                # 結果顯示區域
                gr.Markdown("### 📹 生成結果")
                output_video = gr.Video(
                    label="生成的影片",
                    height=400
                )
                generation_status = gr.Textbox(
                    label="生成狀態",
                    interactive=False,
                    value="等待開始..."
                )
                # 使用提示和故障排除
                gr.Markdown("""
                ### 💡 使用提示
                **最佳效果的圖片特徵：**
                - 清晰度高，對比鮮明
                - 主體居中，背景簡潔
                - 人物或物體邊界清楚
                - 尺寸建議：512x512 像素
                **參數說明：**
                - **動作強度**: 1-50 輕微動作，50-150 中等動作，150+ 強烈動作
                - **隨機性**: 0.0 最穩定，1.0 最多變
                - **隨機種子**: 固定數值可重複生成相似結果
                """)
        # 預設範例
        gr.Markdown("---")
        gr.Markdown("### 🎯 快速開始範例")
        with gr.Row():
            example_btn1 = gr.Button("人像動畫 (溫和)", size="sm")
            example_btn2 = gr.Button("風景動畫 (中等)", size="sm")
            example_btn3 = gr.Button("創意動畫 (強烈)", size="sm")
        # 頁腳
        gr.Markdown("""
        ---
        <div class="footer">
        🤖 Powered by Local Model | 
        Made with ❤ using Gradio
        </div>
        """)
        # 事件綁定
        generate_button.click(
            fn=generate_video,
            inputs=[input_image, motion_strength, randomness, seed_value],
            outputs=[output_video, generation_status]
        )
        # 範例按鈕事件
        example_btn1.click(
            lambda: (50, 0.02, 42),
            outputs=[motion_strength, randomness, seed_value]
        )
        example_btn2.click(
            lambda: (127, 0.05, 123),
            outputs=[motion_strength, randomness, seed_value]
        )
        example_btn3.click(
            lambda: (200, 0.1, 456),
            outputs=[motion_strength, randomness, seed_value]
        )
    return demo

# 主程式
if __name__ == "__main__":
    try:
        import gradio as gr
    except ImportError:
        print("❌ 請先安裝 Gradio:")
        print("pip install gradio")
        exit(1)
    print("🚀 啟動圖片生成影片介面...")
    demo = create_gradio_interface()
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        debug=True
    )
