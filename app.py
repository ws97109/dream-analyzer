import os
import time
import uuid
import json
import random
from flask import Flask, request, jsonify, render_template, url_for
import requests
from PIL import Image
import numpy as np
import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler, StableVideoDiffusionPipeline
import cv2
import gc

class VideoGenerator:
    """獨立的影片生成類別"""
    
    def __init__(self, device=None, torch_dtype=None):
        self.video_pipe = None
        self.device = device or "cpu"
        self.torch_dtype = torch_dtype or torch.float32
        self.video_loaded = False
        
    def load_video_model(self):
        """載入影片生成模型"""
        if self.video_loaded:
            return True
            
        try:
            print("🎬 載入 Stable Video Diffusion 模型...")
            video_model_id = "stabilityai/stable-video-diffusion-img2vid-xt"
            
            video_kwargs = {
                "torch_dtype": self.torch_dtype
            }
            
            # 針對不同設備的優化
            if self.device != "mps" and self.torch_dtype == torch.float16:
                video_kwargs["variant"] = "fp16"
            
            self.video_pipe = StableVideoDiffusionPipeline.from_pretrained(
                video_model_id, **video_kwargs
            ).to(self.device)
            
            # 記憶體優化設定
            if self.device == "cuda":
                self.video_pipe.enable_model_cpu_offload()
                self.video_pipe.enable_vae_slicing()
            elif self.device == "mps":
                self.video_pipe.enable_attention_slicing(1)
            else:
                self.video_pipe.enable_attention_slicing()
            
            self.video_loaded = True
            print(f"✅ 影片生成模型載入完成，使用設備: {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ 影片模型載入失敗: {e}")
            self.video_pipe = None
            self.video_loaded = False
            return False
    
    def generate_video_from_image(self, image_path, static_dir, story_text=""):
        """從圖像生成影片"""
        try:
            # 確保模型已載入
            if not self.load_video_model() or self.video_pipe is None:
                print("❌ 影片生成模型未載入")
                return None
            
            # 載入圖像
            full_image_path = os.path.join(static_dir, image_path)
            if not os.path.exists(full_image_path):
                print(f"❌ 找不到圖像文件: {full_image_path}")
                return None
            
            input_image = Image.open(full_image_path)
            
            # 確保圖像為RGB模式
            if input_image.mode != 'RGB':
                input_image = input_image.convert('RGB')
            
            # 調整圖像尺寸（SVD 需要特定尺寸比例）
            target_width, target_height = 1024, 576
            input_image = input_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            print("🎬 開始生成影片...")
            
            # 生成參數（針對速度優化）
            video_params = {
                "image": input_image,
                "decode_chunk_size": 2,  # 較小的chunk size減少記憶體使用
                "generator": torch.Generator(device=self.device).manual_seed(42),
                "motion_bucket_id": 127,  # 中等運動強度
                "noise_aug_strength": 0.02,  # 較低的噪聲增強以提高穩定性
                "num_frames": 25,  # 標準幀數
            }
            
            # 針對不同設備調整參數
            if self.device == "cpu":
                video_params["num_frames"] = 14  # CPU模式減少幀數
                video_params["decode_chunk_size"] = 1
            elif self.device == "mps":
                video_params["decode_chunk_size"] = 4
            
            start_time = time.time()
            
            # 清理記憶體
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            with torch.no_grad():
                frames = self.video_pipe(**video_params).frames[0]
            
            generation_time = time.time() - start_time
            print(f"🎬 影片幀生成完成，耗時: {generation_time:.2f}秒，幀數: {len(frames)}")
            
            if not frames or len(frames) == 0:
                print("❌ 影片幀生成失敗")
                return None
            
            # 保存影片
            timestamp = int(time.time())
            random_id = str(uuid.uuid4())[:8]
            output_filename = f"dream_video_{timestamp}_{random_id}.mp4"
            
            output_dir = os.path.join(static_dir, 'videos')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, output_filename)
            
            # 使用 OpenCV 保存影片
            fps = 8  # 8 FPS
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
            
            for frame in frames:
                if frame is None:
                    continue
                frame_array = np.array(frame)
                if frame_array.shape[2] == 3:  # RGB
                    frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame_array
                out.write(frame_bgr)
            
            out.release()
            
            # 驗證影片文件
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"✅ 影片保存成功: {output_filename}")
                return os.path.join('videos', output_filename)
            else:
                print("❌ 影片文件保存失敗或文件為空")
                return None
                
        except Exception as e:
            print(f"❌ 影片生成失敗: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def clear_video_memory(self):
        """清理影片模型記憶體"""
        if self.video_pipe is not None:
            del self.video_pipe
            self.video_pipe = None
        
        self.video_loaded = False
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        print("🎬 影片模型記憶體已清理")


class DreamAnalyzer:
    def __init__(self):
        self.app_root = os.path.dirname(os.path.abspath(__file__))
        self.static_dir = os.path.join(self.app_root, 'static')
        self.app = Flask(__name__, 
                        template_folder=os.path.join(self.app_root, 'templates'),
                        static_folder=self.static_dir)
        
        # 配置
        self.OLLAMA_API = "http://localhost:11434/api/generate"
        self.OLLAMA_MODEL = "qwen2.5:14b"
        
        # 模型狀態
        self.image_pipe = None
        self.models_loaded = False
        self.current_device = None
        self.torch_dtype = None
        
        # 影片生成器
        self.video_generator = None
        
        # 可用模型
        self.models = {
            "stable-diffusion-v1-5": "runwayml/stable-diffusion-v1-5",
        }
        
        # 防重複提交
        self.processing_requests = set()  # 正在處理的請求ID
        self.request_lock = False  # 全局鎖
        
        self._setup_routes()
        self._create_directories()

    def _create_directories(self):
        """創建必要的目錄"""
        directories = [
            os.path.join(self.static_dir, 'images'),
            os.path.join(self.static_dir, 'videos'),  # 新增影片目錄
            os.path.join(self.static_dir, 'shares')
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def _setup_routes(self):
        """設定路由"""
        self.app.route('/')(self.index)
        self.app.route('/api/status')(self.api_status)
        self.app.route('/api/analyze', methods=['POST'])(self.analyze)
        self.app.route('/api/share', methods=['POST'])(self.share_result)
        self.app.route('/share/<share_id>')(self.view_shared)

    def _initialize_device(self):
        """初始化設備設定"""
        if torch.cuda.is_available():
            self.current_device = "cuda"
            self.torch_dtype = torch.float16
        elif torch.backends.mps.is_available():
            self.current_device = "mps"
            self.torch_dtype = torch.float32
        else:
            self.current_device = "cpu"
            self.torch_dtype = torch.float32

    def _initialize_video_generator(self):
        """初始化影片生成器"""
        if self.video_generator is None:
            self._initialize_device()
            self.video_generator = VideoGenerator(
                device=self.current_device, 
                torch_dtype=self.torch_dtype
            )
        return self.video_generator

    def _load_image_model(self):
        """載入圖像生成模型"""
        if self.models_loaded:
            return True
        
        try:
            self._initialize_device()
            model_id = "runwayml/stable-diffusion-v1-5"
            
            self.image_pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                use_safetensors=True,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.current_device)
            
            self.image_pipe.scheduler = UniPCMultistepScheduler.from_config(
                self.image_pipe.scheduler.config
            )
            
            # 優化設定
            if self.current_device == "cuda":
                self.image_pipe.enable_model_cpu_offload()
                self.image_pipe.enable_attention_slicing()
                self.image_pipe.enable_vae_slicing()
            elif self.current_device == "mps":
                self.image_pipe.enable_attention_slicing(1)
            else:
                self.image_pipe.enable_attention_slicing()
            
            self.models_loaded = True
            print(f"✅ 模型載入成功，使用設備: {self.current_device}")
            return True
            
        except Exception as e:
            print(f"❌ 模型載入失敗: {e}")
            return False

    def _check_ollama_status(self):
        """檢查 Ollama 服務狀態"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def _call_ollama(self, system_prompt, user_prompt, temperature=0.7):
        """調用 Ollama API"""
        try:
            data = {
                "model": self.OLLAMA_MODEL,
                "prompt": user_prompt,
                "system": system_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": 0.9,
                    "num_predict": 500
                }
            }
            
            response = requests.post(self.OLLAMA_API, json=data, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            return ""
        except Exception as e:
            print(f"❌ Ollama 調用失敗: {e}")
            return ""

    def _generate_story(self, dream_text):
        """生成夢境故事"""
        system_prompt = """你是夢境故事創作專家，要求：
1. 直接開始故事，無問候語
2. 融合夢境元素
3. 使用第一人稱
4. 150-200字完整故事
5. 繁體中文"""

        user_prompt = f"基於夢境片段創作故事：「{dream_text}」"
        
        story = self._call_ollama(system_prompt, user_prompt)
        return self._clean_story_content(story) if story else "無法生成夢境故事"

    def _clean_story_content(self, story):
        """清理故事內容"""
        unwanted_phrases = [
            "好的，根據您的建議", "根據您的要求", "以下是故事", "故事如下",
            "###", "**", "故事名稱：", "夢境故事：", "完整故事："
        ]
        
        cleaned_story = story.strip()
        
        for phrase in unwanted_phrases:
            if cleaned_story.startswith(phrase):
                cleaned_story = cleaned_story[len(phrase):].strip()
            if phrase in cleaned_story:
                parts = cleaned_story.split(phrase)
                if len(parts) > 1:
                    cleaned_story = parts[-1].strip()
        
        # 移除引號
        if cleaned_story.startswith('"') and cleaned_story.endswith('"'):
            cleaned_story = cleaned_story[1:-1].strip()
        if cleaned_story.startswith('「') and cleaned_story.endswith('」'):
            cleaned_story = cleaned_story[1:-1].strip()
        
        return cleaned_story.replace('*', '').replace('#', '').strip()

    def _generate_image_prompt(self, dream_text):
        """將夢境轉換為圖像生成提示詞"""
        system_prompt = """You are a Stable Diffusion prompt expert. Convert user input into English image generation prompts. Requirements:
        1. PRESERVE and translate ALL original elements from the input
        2. If input is a story/emotion, extract the MOST VISUAL and EMOTIONAL scene
        3. ENHANCE with additional quality details, don't replace original content
        4. Use English keywords only
        5. The main characters are mostly Asian
        6. Focus on the most dramatic or emotional moment in the story
        7. Include facial expressions and emotions from the original text

        Example:
        Input: "他常夢見前任結婚了，每次都笑著祝福，然後哭著醒來"
        Better output: "Asian man dreaming, wedding scene, forced smile, tears, emotional pain, bittersweet expression, dream-like atmosphere, cinematic lighting, detailed"

        Always capture the EMOTIONAL CORE and most visual elements."""
        user_prompt = f"Story: {dream_text}\nConvert to image prompt:"
        
        raw_prompt = self._call_ollama(system_prompt, user_prompt, temperature=0.5)
        
        if not raw_prompt:
            return None
        
        # 清理並處理提示詞
        clean_prompt = raw_prompt.strip()
        
        # 檢查是否包含人物元素
        if ('我' in dream_text or '自己' in dream_text) and \
           not any(word in clean_prompt.lower() for word in ['person', 'human', 'figure']):
            clean_prompt = f" I {clean_prompt}"
        
        final_prompt = f"{clean_prompt}, vibrant colors, detailed, cinematic"
        
        print(f"✨ 生成圖像提示詞: {final_prompt}")
        return final_prompt

    def _generate_image(self, dream_text):
        """生成圖像"""
        if not self._load_image_model() or self.image_pipe is None:
            print("❌ 圖像模型未載入")
            return None
        
        try:
            # 生成提示詞
            image_prompt = self._generate_image_prompt(dream_text)
            if not image_prompt:
                print("❌ 無法生成圖像提示詞")
                return None
            
            # 設定負面提示詞
            person_keywords = ['我', '人', '自己', '夢見我']
            if any(keyword in dream_text for keyword in person_keywords):
                negative_prompt = "ugly, blurry, distorted, deformed, low quality, bad anatomy, multiple heads, extra limbs"
            else:
                negative_prompt = "human, person, ugly, blurry, distorted, deformed, low quality, bad anatomy"
            
            # 生成參數
            seed = random.randint(0, 2**32 - 1)
            generator = torch.Generator(self.current_device).manual_seed(seed)
            
            generation_params = {
                "prompt": image_prompt,
                "negative_prompt": negative_prompt,
                "height": 512,
                "width": 512,
                "num_inference_steps": 20 if self.current_device != "cpu" else 15,
                "guidance_scale": 7.5,
                "generator": generator
            }
            
            # 清理記憶體
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # 生成圖像
            print("🎨 開始生成圖像...")
            with torch.no_grad():
                result = self.image_pipe(**generation_params)
                
                if not result.images or len(result.images) == 0:
                    print("❌ 圖像生成失敗")
                    return None
                
                generated_image = result.images[0]
            
            # 處理圖像
            if generated_image.mode != 'RGB':
                generated_image = generated_image.convert('RGB')
            
            # 調整亮度
            img_array = np.array(generated_image)
            avg_brightness = np.mean(img_array)
            
            if avg_brightness < 30:
                img_array = np.clip(img_array * 1.3 + 20, 0, 255).astype(np.uint8)
                generated_image = Image.fromarray(img_array)
            
            # 保存圖像
            timestamp = int(time.time())
            random_id = str(uuid.uuid4())[:8]
            output_filename = f"dream_{timestamp}_{random_id}.png"
            
            output_dir = os.path.join(self.static_dir, 'images')
            output_path = os.path.join(output_dir, output_filename)
            
            generated_image.save(output_path, format='PNG', quality=95)
            print(f"✅ 圖像已保存: {output_filename}")
            
            return os.path.join('images', output_filename)
            
        except Exception as e:
            print(f"❌ 圖像生成錯誤: {e}")
            return None

    def _generate_video(self, image_path, dream_text):
        """生成影片（新增功能）"""
        try:
            # 初始化影片生成器
            video_gen = self._initialize_video_generator()
            
            # 生成影片
            video_path = video_gen.generate_video_from_image(
                image_path, self.static_dir, dream_text
            )
            
            return video_path
            
        except Exception as e:
            print(f"❌ 影片生成錯誤: {e}")
            return None

    def _analyze_psychology(self, dream_text):
        """心理分析"""
        system_prompt = """你是夢境心理分析專家，分析夢境的象徵意義和心理狀態，
提供150-200字的分析，使用溫和支持性語調，繁體中文回答。"""
        
        user_prompt = f"夢境描述: {dream_text}\n請提供心理分析："
        
        analysis = self._call_ollama(system_prompt, user_prompt)
        return analysis if analysis else "暫時無法進行心理分析。"

    def _save_dream_result(self, data):
        """保存夢境分析結果"""
        try:
            share_id = str(uuid.uuid4())
            share_dir = os.path.join(self.static_dir, 'shares')
            
            share_data = {
                'id': share_id,
                'timestamp': int(time.time()),
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'finalStory': data.get('finalStory', ''),
                'imagePath': data.get('imagePath', ''),
                'videoPath': data.get('videoPath', ''),  # 新增影片路徑
                'psychologyAnalysis': data.get('psychologyAnalysis', '')
            }
            
            share_file = os.path.join(share_dir, f"{share_id}.json")
            with open(share_file, 'w', encoding='utf-8') as f:
                json.dump(share_data, f, ensure_ascii=False, indent=2)
            
            return share_id
        except Exception as e:
            print(f"❌ 保存分享結果失敗: {e}")
            return None

    # 路由處理函數
    def index(self):
        return render_template('index.html')

    def api_status(self):
        ollama_status = self._check_ollama_status()
        local_models_status = self.models_loaded or self._load_image_model()
        
        # 檢查影片生成器狀態
        video_status = False
        if self.video_generator:
            video_status = self.video_generator.video_loaded
        
        return jsonify({
            'ollama': ollama_status,
            'local_models': local_models_status,
            'video_models': video_status,  # 新增影片模型狀態
            'device': self.current_device,
            'available_models': list(self.models.keys()),
            'timestamp': int(time.time())
        })

    def analyze(self):
        data = request.json
        dream_text = data.get('dream', '')
        selected_model = data.get('model', 'stable-diffusion-v1-5')
        generate_video = data.get('generateVideo', False)  # 新增影片生成選項
        
        # 輸入驗證
        if not dream_text or len(dream_text.strip()) < 10:
            return jsonify({'error': '請輸入至少10個字的夢境描述'}), 400
        
        if len(dream_text.strip()) > 2000:
            return jsonify({'error': '夢境描述過長，請控制在2000字以內'}), 400
        
        # 防重複提交檢查
        request_id = f"{dream_text[:50]}_{int(time.time())}"
        
        if self.request_lock:
            print("⚠️  有其他請求正在處理中，請稍候...")
            return jsonify({'error': '系統正在處理其他請求，請稍候再試'}), 429
        
        if request_id in self.processing_requests:
            print("⚠️  相同請求已在處理中...")
            return jsonify({'error': '相同的請求正在處理中'}), 429
        
        # 設定處理狀態
        self.request_lock = True
        self.processing_requests.add(request_id)
        
        try:
            print(f"🌙 開始分析夢境 [ID: {request_id[:20]}...]: {dream_text[:50]}...")
            
            # 檢查服務狀態
            ollama_status = self._check_ollama_status()
            if not ollama_status:
                return jsonify({'error': 'Ollama服務不可用'}), 503
            
            # 生成故事
            print("📖 生成夢境故事...")
            final_story = self._generate_story(dream_text)
            
            # 生成圖像
            print("🎨 生成夢境圖像...")
            local_models_status = self._load_image_model()
            image_path = None
            if local_models_status:
                image_path = self._generate_image(dream_text)
            
            # 生成影片（新增功能）
            video_path = None
            if generate_video and image_path and local_models_status:
                print("🎬 生成夢境影片...")
                video_path = self._generate_video(image_path, dream_text)
                if video_path:
                    print("✅ 影片生成成功")
                else:
                    print("⚠️  影片生成失敗，但不影響其他功能")
            
            # 心理分析
            print("🧠 進行心理分析...")
            psychology_analysis = self._analyze_psychology(dream_text)
            
            response = {
                'finalStory': final_story,
                'imagePath': '/static/' + image_path if image_path else None,
                'videoPath': '/static/' + video_path if video_path else None,  # 新增影片路徑
                'psychologyAnalysis': psychology_analysis,
                'apiStatus': {
                    'ollama': ollama_status,
                    'local_models': local_models_status,
                    'video_models': self.video_generator.video_loaded if self.video_generator else False,  # 新增影片模型狀態
                    'device': self.current_device,
                    'current_model': selected_model
                },
                'processingInfo': {
                    'timestamp': int(time.time()),
                    'inputLength': len(dream_text),
                    'storyLength': len(final_story) if final_story else 0,
                    'requestId': request_id[:20],
                    'videoGenerated': video_path is not None  # 新增影片生成狀態
                }
            }
            
            print(f"✅ 夢境分析完成 [ID: {request_id[:20]}...]")
            return jsonify(response)
            
        except Exception as e:
            print(f"❌ 分析錯誤 [ID: {request_id[:20]}...]: {e}")
            return jsonify({'error': '處理過程中發生錯誤'}), 500
        
        finally:
            # 清理處理狀態
            self.request_lock = False
            self.processing_requests.discard(request_id)
            print(f"🔓 釋放請求鎖 [ID: {request_id[:20]}...]")

    def share_result(self):
        data = request.json
        
        if not data or 'finalStory' not in data:
            return jsonify({'error': '缺少必要的夢境分析數據'}), 400
        
        try:
            share_id = self._save_dream_result(data)
            
            if not share_id:
                return jsonify({'error': '創建分享失敗'}), 500
            
            share_url = url_for('view_shared', share_id=share_id, _external=True)
            
            return jsonify({
                'shareId': share_id, 
                'shareUrl': share_url,
                'timestamp': int(time.time())
            })
            
        except Exception as e:
            print(f"❌ 分享處理錯誤: {e}")
            return jsonify({'error': '處理分享請求時發生錯誤'}), 500

    def view_shared(self, share_id):
        try:
            share_file = os.path.join(self.static_dir, 'shares', f"{share_id}.json")
            
            if not os.path.exists(share_file):
                return jsonify({'error': '找不到該分享內容'}), 404
            
            with open(share_file, 'r', encoding='utf-8') as f:
                share_data = json.load(f)
            
            return render_template('shared.html', data=share_data)
            
        except Exception as e:
            print(f"❌ 載入分享內容錯誤: {e}")
            return jsonify({'error': '載入分享內容時發生錯誤'}), 500

    def clear_all_memory(self):
        """清理所有模型記憶體"""
        # 清理圖像模型
        if self.image_pipe is not None:
            del self.image_pipe
            self.image_pipe = None
        
        # 清理影片模型
        if self.video_generator:
            self.video_generator.clear_video_memory()
        
        self.models_loaded = False
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        print("🧹 所有模型記憶體已清理")

    def run(self, debug=False, host='0.0.0.0', port=5002):
        """啟動應用"""
        try:
            # 在程式結束時清理記憶體
            import atexit
            atexit.register(self.clear_all_memory)
            
            print("🚀 啟動夢境分析應用...")
            
            # 檢查服務狀態
            print("檢查服務狀態...")
            ollama_status = self._check_ollama_status()
            local_models_status = self._load_image_model()
            
            # 輸出狀態報告
            print("=" * 80)
            print("夢境分析系統 - 含影片生成功能 啟動狀態報告")
            print("=" * 80)
            print(f"Ollama API (localhost:11434): {'✅ 正常' if ollama_status else '❌ 異常'}")
            print(f"本地圖像生成模型: {'✅ 可用' if local_models_status else '❌ 不可用'}")
            print(f"本地影片生成模型: {'✅ 可用' if self.video_generator and self.video_generator.video_loaded else '⚠️  需要時載入'}")
            print(f"靜態檔案目錄: {self.static_dir}")
            
            # 檢查 PyTorch 和設備支持
            if torch.backends.mps.is_available():
                print("✅ 已啟用 Metal Performance Shaders (MPS) 加速")
                device_info = "MPS (Apple Silicon 優化)"
            elif torch.cuda.is_available():
                print("✅ 已啟用 CUDA 加速")
                device_info = f"CUDA - {torch.cuda.get_device_name()}"
            else:
                print("⚠️  使用 CPU 模式，速度可能較慢")
                device_info = "CPU"
            
            print(f"PyTorch 版本: {torch.__version__}")
            print(f"使用設備: {device_info}")
            print("=" * 80)
            
            # 系統功能說明
            print("🔧 系統功能狀態:")
            print(f"   • 故事生成: {'✅ 可用 (Ollama qwen2.5:14b)' if ollama_status else '❌ 不可用'}")
            print(f"   • 圖像生成: {'✅ 可用 (Stable Diffusion v1.5)' if local_models_status else '❌ 不可用'}")
            print(f"   • 影片生成: {'✅ 可用 (Stable Video Diffusion)' if self.video_generator else '⚠️  需要時載入'}")
            print(f"   • 心理分析: {'✅ 可用 (Ollama)' if ollama_status else '❌ 不可用'}")
            print()
            
            print("🎬 影片生成特性:")
            print("   • 圖像轉影片: 從生成的圖像創建動態影片")
            print("   • 解析度: 1024x576，25幀 (CPU模式為14幀)")
            print("   • 幀率: 8 FPS")
            print("   • 格式: MP4")
            print("   • 智能記憶體管理")
            print("   • 可選擇是否生成影片")
            print()
            
            if not ollama_status:
                print("❌ 警告: Ollama API 無法連接")
                print("   請確認 Ollama 服務是否運行在 localhost:11434")
                print("   啟動命令: ollama serve")
                print("   必須先安裝模型: ollama pull qwen2.5:14b")
                print()
            
            if not local_models_status:
                print("❌ 警告: 本地生成模型不可用")
                print("   首次運行時會自動下載模型")
                print("   圖像模型約 4GB，影片模型約 6-8GB")
                print("   請確保網路連接正常且有足夠的儲存空間")
                print()
            
            print("⚡ 性能預期:")
            if device_info.startswith("MPS"):
                print("   • Apple Silicon 優化")
                print("   • 圖像生成: 10-30 秒")
                print("   • 影片生成: 1-3 分鐘")
                print("   • 建議: 16GB+ 統一記憶體")
            elif device_info.startswith("CUDA"):
                print("   • GPU 加速，速度最快")
                print("   • 圖像生成: 5-15 秒")
                print("   • 影片生成: 30秒-2分鐘")
                print("   • 建議: 8GB+ VRAM")
            else:
                print("   • CPU 模式，速度較慢")
                print("   • 圖像生成: 1-3 分鐘")
                print("   • 影片生成: 5-10 分鐘")
                print("   • 建議: 16GB+ RAM")
            print()
            
            print("🆕 新增功能:")
            print("   • 獨立的影片生成模組")
            print("   • 可選擇是否生成影片")
            print("   • 影片生成狀態監控")
            print("   • 智能記憶體清理")
            print("   • 分享功能包含影片")
            print()
            
            print("=" * 80)
            print("系統準備就緒，啟動 Flask 應用程式...")
            print("訪問地址: http://localhost:5002")
            print("=" * 80)
            
            self.app.run(debug=debug, host=host, port=port, threaded=True)
            
        except KeyboardInterrupt:
            print("用戶中斷，正在關閉系統...")
            self.clear_all_memory()
        except Exception as e:
            print(f"系統啟動失敗: {e}")
            import traceback
            traceback.print_exc()
            self.clear_all_memory()
        finally:
            print("正在清理系統資源...")
            self.clear_all_memory()


# 主程式入口
if __name__ == '__main__':
    dream_analyzer = DreamAnalyzer()
    dream_analyzer.run(debug=False)
