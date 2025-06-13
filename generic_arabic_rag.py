import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import re
import torch
from collections import Counter
import spacy
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class ImprovedArabicRAGQuestionGenerator:
    def __init__(self):
        """
        Version améliorée du générateur de questions arabes
        """
        # Initialiser les patterns de domaine en premier (pour éviter AttributeError)
        self.domain_patterns = {
            'science': {
                'keywords': ['تجربة', 'عملية', 'نتيجة', 'تفاعل', 'مادة', 'طاقة', 'خلية', 'ذرة', 'جزيء'],
                'question_starters': ['كيف تتم', 'لماذا يحدث', 'ما هي عملية', 'كيف يؤثر']
            },
            'history': {
                'keywords': ['حدث', 'تاريخ', 'ملك', 'حرب', 'معركة', 'دولة', 'حضارة', 'عصر', 'فترة'],
                'question_starters': ['متى حدث', 'من هو', 'كيف أثر', 'لماذا وقع']
            },
            'literature': {
                'keywords': ['قصة', 'شاعر', 'كاتب', 'رواية', 'قصيدة', 'أدب', 'نص', 'ديوان'],
                'question_starters': ['من كتب', 'ما موضوع', 'كيف يعبر', 'ما مضمون']
            },
            'geography': {
                'keywords': ['مدينة', 'بلد', 'جبل', 'نهر', 'صحراء', 'مناخ', 'موقع', 'إقليم'],
                'question_starters': ['أين يقع', 'ما مناخ', 'كيف يتميز', 'ما خصائص']
            },
            'religion': {
                'keywords': ['الله', 'قرآن', 'حديث', 'صلاة', 'زكاة', 'حج', 'صوم', 'إسلام'],
                'question_starters': ['ما حكم', 'كيف نؤدي', 'ما فضل', 'متى يكون']
            },
            'general': {
                'keywords': ['معلومة', 'بيانات', 'حقيقة', 'موضوع', 'قضية'],
                'question_starters': ['ما هو', 'كيف يمكن', 'لماذا', 'متى']
            }
        }
        
        # Variables d'état
        self.index = None
        self.segments = []
        self.embeddings = None
        self.is_initialized = False
        
        try:
            # Modèle pour les embeddings - utiliser un modèle plus fiable
            print("تحميل نموذج التضمينات...")
            self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            
            # Essayer plusieurs modèles arabes alternatifs
            models_to_try = [
                'CAMeL-Lab/bert-base-arabic-camelbert-mix',
                'aubmindlab/bert-base-arabertv2',
                'microsoft/DialoGPT-medium',  # Modèle de génération générique
                'facebook/mbart-large-50-many-to-many-mmt'  # Modèle multilingue
            ]
            
            self.tokenizer = None
            self.generator_model = None
            self.text_generator = None
            
            # Essayer de charger un modèle disponible
            for model_name in models_to_try:
                try:
                    print(f"محاولة تحميل النموذج: {model_name}")
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    
                    # Vérifier si c'est un modèle de génération de texte
                    if 'mbart' in model_name.lower() or 't5' in model_name.lower():
                        self.generator_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                        
                        # Pipeline de génération
                        self.text_generator = pipeline(
                            "text2text-generation",
                            model=self.generator_model,
                            tokenizer=self.tokenizer,
                            max_length=200,
                            num_return_sequences=1,
                            temperature=0.7
                        )
                    
                    print(f"✓ تم تحميل النموذج بنجاح: {model_name}")
                    break
                    
                except Exception as e:
                    print(f"فشل في تحميل {model_name}: {e}")
                    continue
            
            # إذا فشل تحميل جميع النماذج، استخدم النظام الأساسي فقط
            if not self.tokenizer:
                print("تحذير: لم يتم تحميل نموذج التوليد، سيتم استخدام النظام الأساسي فقط")
            
            self.is_initialized = True
            print("✓ تم تهيئة النظام بنجاح")
            
        except Exception as e:
            print(f"خطأ في التهيئة: {e}")
            self.is_initialized = False
    
    def extract_key_phrases(self, text: str, max_phrases: int = 5) -> List[str]:
        """
        استخراج العبارات المفتاحية بذكاء أكبر
        """
        # تنظيف النص
        cleaned_text = re.sub(r'[^\w\s]', ' ', text)
        words = cleaned_text.split()
        
        # استخراج العبارات الثنائية والثلاثية
        phrases = []
        
        # عبارات ثنائية
        for i in range(len(words) - 1):
            if len(words[i]) > 2 and len(words[i+1]) > 2:
                phrase = f"{words[i]} {words[i+1]}"
                phrases.append(phrase)
        
        # عبارات ثلاثية
        for i in range(len(words) - 2):
            if all(len(word) > 2 for word in words[i:i+3]):
                phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
                phrases.append(phrase)
        
        # حساب تكرار العبارات
        phrase_counts = Counter(phrases)
        
        # إرجاع أهم العبارات
        return [phrase for phrase, count in phrase_counts.most_common(max_phrases)]
    
    def extract_key_entities(self, text: str) -> Dict[str, List[str]]:
        """
        استخراج الكيانات المسماة من النص
        """
        entities = {
            'persons': [],
            'locations': [],
            'organizations': [],
            'concepts': []
        }
        
        # مؤشرات بسيطة للأشخاص
        person_patterns = [r'الدكتور\s+\w+', r'الأستاذ\s+\w+', r'الشيخ\s+\w+']
        for pattern in person_patterns:
            matches = re.findall(pattern, text)
            entities['persons'].extend(matches)
        
        # مؤشرات بسيطة للأماكن
        location_patterns = [r'مدينة\s+\w+', r'بلد\s+\w+', r'في\s+\w+']
        for pattern in location_patterns:
            matches = re.findall(pattern, text)
            entities['locations'].extend(matches)
        
        return entities
    
    def generate_contextual_questions(self, segment: str, domain: str) -> Dict[str, str]:
        """
        توليد أسئلة ذكية باستخدام السياق والذكاء الاصطناعي
        """
        # استخراج العبارات المفتاحية من النص
        key_phrases = self.extract_key_phrases(segment)
        
        # تحليل النص لفهم المحتوى
        content_analysis = self.analyze_content_structure(segment)
        
        questions = {}
        
        # سؤال مباشر مبني على المحتوى الفعلي
        if key_phrases:
            main_concept = key_phrases[0]
            
            # تحديد نوع السؤال بناء على تحليل المحتوى
            if content_analysis['has_process']:
                questions['direct'] = f"كيف تتم {main_concept}؟"
            elif content_analysis['has_definition']:
                questions['direct'] = f"ما تعريف {main_concept}؟"
            elif content_analysis['has_location']:
                questions['direct'] = f"أين تحدث/توجد {main_concept}؟"
            elif content_analysis['has_person']:
                questions['direct'] = f"من هو المسؤول عن {main_concept}؟"
            else:
                questions['direct'] = f"ما هو {main_concept}؟"
        else:
            questions['direct'] = "ما الفكرة الرئيسية في هذا النص؟"
        
        # سؤال تفسيري يتطلب تحليل أعمق
        if content_analysis['has_cause_effect']:
            questions['explanatory'] = "لماذا تحدث هذه النتائج وما العوامل المؤثرة؟"
        elif content_analysis['has_comparison']:
            questions['explanatory'] = "كيف تقارن بين العناصر المختلفة المذكورة؟"
        elif content_analysis['has_sequence']:
            questions['explanatory'] = "ما التسلسل المنطقي للأحداث أو العمليات المذكورة؟"
        else:
            questions['explanatory'] = f"كيف يمكن تطبيق هذه المعلومات في السياق العملي؟"
        
        return questions
    
    def analyze_content_structure(self, text: str) -> Dict[str, bool]:
        """
        تحليل بنية المحتوى لفهم نوع المعلومات
        """
        analysis = {
            'has_process': False,
            'has_definition': False,
            'has_location': False,
            'has_person': False,
            'has_cause_effect': False,
            'has_comparison': False,
            'has_sequence': False,
            'has_numbers': False
        }
        
        # البحث عن مؤشرات العمليات
        process_indicators = ['تتم', 'يحدث', 'تحدث', 'عملية', 'طريقة', 'أسلوب']
        analysis['has_process'] = any(indicator in text for indicator in process_indicators)
        
        # البحث عن مؤشرات التعريفات
        definition_indicators = ['هو', 'هي', 'يعرف', 'تعريف', 'مفهوم', 'يقصد']
        analysis['has_definition'] = any(indicator in text for indicator in definition_indicators)
        
        # البحث عن مؤشرات المكان
        location_indicators = ['في', 'من', 'إلى', 'عند', 'لدى', 'موقع', 'مكان']
        analysis['has_location'] = any(indicator in text for indicator in location_indicators)
        
        # البحث عن مؤشرات الأشخاص
        person_indicators = ['الذي', 'التي', 'من', 'شخص', 'رجل', 'امرأة']
        analysis['has_person'] = any(indicator in text for indicator in person_indicators)
        
        # البحث عن مؤشرات السبب والنتيجة
        cause_effect_indicators = ['لأن', 'بسبب', 'نتيجة', 'لذلك', 'فإن', 'يؤدي إلى']
        analysis['has_cause_effect'] = any(indicator in text for indicator in cause_effect_indicators)
        
        # البحث عن مؤشرات المقارنة
        comparison_indicators = ['أكثر', 'أقل', 'مثل', 'مقارنة', 'خلافا', 'بينما', 'أما']
        analysis['has_comparison'] = any(indicator in text for indicator in comparison_indicators)
        
        # البحث عن مؤشرات التسلسل
        sequence_indicators = ['أولا', 'ثانيا', 'ثم', 'بعد ذلك', 'أخيرا', 'في البداية']
        analysis['has_sequence'] = any(indicator in text for indicator in sequence_indicators)
        
        # البحث عن الأرقام
        analysis['has_numbers'] = bool(re.search(r'\d+|[٠-٩]+', text))
        
        return analysis
    
    def process_text_and_generate_questions(self, arabic_text: str, max_questions: int = 3) -> List[Dict]:
        """
        معالجة النص وتوليد أسئلة ذكية
        """
        if not self.is_initialized:
            return []
        
        try:
            # تقسيم النص إلى فقرات
            segments = self.preprocess_arabic_text(arabic_text)
            self.segments = segments  # حفظ الفقرات
            
            if not segments:
                return []
            
            # تحديد المجال
            domain = self.detect_text_domain(arabic_text)
            
            # اختيار أفضل الفقرات
            best_segments = self.select_best_segments(segments, max_questions)
            
            results = []
            for i, segment in enumerate(best_segments, 1):
                # توليد أسئلة ذكية لكل فقرة
                questions = self.generate_contextual_questions(segment, domain)
                
                results.append({
                    'question_number': i,
                    'segment': segment,
                    'direct': questions['direct'],
                    'explicit': questions['explanatory'],
                    'domain': domain
                })
            
            return results
            
        except Exception as e:
            print(f"خطأ في المعالجة: {e}")
            return []
    
    def select_best_segments(self, segments: List[str], count: int) -> List[str]:
        """
        اختيار أفضل الفقرات للأسئلة
        """
        scored_segments = []
        
        for segment in segments:
            score = 0
            
            # تفضيل الجمل متوسطة الطول
            if 30 <= len(segment) <= 150:
                score += 3
            elif len(segment) > 150:
                score += 1
            
            # البحث عن كلمات مفتاحية مهمة
            important_words = ['لأن', 'بسبب', 'نتيجة', 'مثل', 'يعتبر', 'يمكن', 'حيث']
            for word in important_words:
                if word in segment:
                    score += 2
            
            # تجنب الجمل القصيرة جداً أو الفارغة من المعنى
            if len(segment.split()) < 5:
                score -= 5
            
            scored_segments.append((segment, score))
        
        # ترتيب وإرجاع أفضل الفقرات
        scored_segments.sort(key=lambda x: x[1], reverse=True)
        return [segment for segment, score in scored_segments[:count]]
    
    def preprocess_arabic_text(self, text: str) -> List[str]:
        """
        معالجة وتقسيم النص العربي
        """
        # تنظيف النص
        text = re.sub(r'\s+', ' ', text.strip())
        
        # تقسيم بناء على علامات الترقيم
        sentences = re.split(r'[.!?؟。]+', text)
        
        # تنظيف وفلترة الجمل
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 15 and len(sentence.split()) > 3:
                clean_sentences.append(sentence)
        
        return clean_sentences
    
    def detect_text_domain(self, text: str) -> str:
        """
        تحديد مجال النص
        """
        domain_scores = {}
        text_lower = text.lower()
        
        for domain, config in self.domain_patterns.items():
            score = 0
            for keyword in config['keywords']:
                score += text_lower.count(keyword)
            domain_scores[domain] = score
        
        best_domain = max(domain_scores, key=domain_scores.get)
        return best_domain if domain_scores[best_domain] > 0 else 'general'

# مثال للاستخدام
def test_improved_generator():
    generator = ImprovedArabicRAGQuestionGenerator()
    
    test_text = """
    يستخدم العلم التكنولوجيا الحديثة، مثل التليسكوبات الفضائية، لدراسة النجوم والمجرات.
    هذا يساعد العلماء على فهم أصل الكون وتطوره عبر مليارات السنين.
    اكتشاف الموجات الثقالية فتح آفاقاً جديدة في فهمنا للكون.
    """
    
    results = generator.process_text_and_generate_questions(test_text, 3)
    
    print("=== النتائج المحسنة ===")
    for result in results:
        print(f"السؤال {result['question_number']}:")
        print(f"النص: {result['segment']}")
        print(f"السؤال المباشر: {result['direct']}")
        print(f"السؤال التفسيري: {result['explicit']}")
        print(f"المجال: {result['domain']}")
        print("-" * 50)

if __name__ == "__main__":
    test_improved_generator()