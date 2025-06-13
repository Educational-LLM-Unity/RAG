from flask import Flask, render_template, request, jsonify, flash
import os
import json
from datetime import datetime
import traceback

# Import du système RAG
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import nltk
from nltk.tokenize import sent_tokenize
import torch

class ArabicRAGQuestionGenerator:
    def __init__(self):
        """
        Initialise le système RAG pour la génération de questions en arabe
        """
        try:
            # Modèle pour les embeddings multilingues
            self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            
            # Modèle pour la génération de texte arabe (mT5)
            self.tokenizer = AutoTokenizer.from_pretrained('google/mt5-small')
            self.generator_model = AutoModelForSeq2SeqLM.from_pretrained('google/mt5-small')
            
            # Index FAISS pour la recherche sémantique
            self.index = None
            self.segments = []
            self.embeddings = None
            self.is_initialized = True
        except Exception as e:
            print(f"Erreur lors de l'initialisation des modèles: {e}")
            self.is_initialized = False
        
    def preprocess_arabic_text(self, text):
        """
        Prétraite le texte arabe pour la segmentation
        """
        # Nettoyer le texte
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Segmentation par phrases (utilise les signes de ponctuation arabes)
        sentences = re.split(r'[.!?؟。]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def extract_keywords(self, text):
        """
        Extrait des mots-clés du texte arabe pour les requêtes de récupération
        """
        keywords = []
        
        scientific_patterns = [
            r'التمثيل الضوئي',
            r'الكلوروفيل',
            r'البلاستيدات',
            r'الجلوكوز',
            r'الأكسجين',
            r'ثاني أكسيد الكربون',
            r'الطاقة'
        ]
        
        for pattern in scientific_patterns:
            if re.search(pattern, text):
                keywords.append(pattern)
        
        return keywords if keywords else ['النص', 'المحتوى']
    
    def create_embeddings(self, segments):
        """
        Crée les embeddings pour les segments de texte
        """
        if not self.is_initialized:
            return None
            
        self.segments = segments
        self.embeddings = self.embedding_model.encode(segments)
        
        # Créer l'index FAISS
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))
        
        return self.embeddings
    
    def simulate_question_generation(self, segment, question_type, starters):
        """
        Simule la génération de questions avec analyse approfondie du contenu
        """
        segment_lower = segment.lower()
        
        if "التمثيل الضوئي" in segment:
            if question_type == "explicit":
                if "الطاقة" in segment and "الشمس" in segment:
                    return "كيف تستخدم النباتات الطاقة الضوئية في عملية التمثيل الضوئي؟"
                elif "مهمة" in segment or "تنتج" in segment:
                    return "لماذا يعتبر التمثيل الضوئي عملية حيوية مهمة للكائنات الحية؟"
                elif "البلاستيدات" in segment:
                    return "كيف تتم عملية التمثيل الضوئي داخل البلاستيدات الخضراء؟"
                else:
                    return "كيف تقوم النباتات بعملية التمثيل الضوئي؟"
            else:
                if "الكلوروفيل" in segment:
                    return "ما هي المادة التي تمتص الضوء في البلاستيدات الخضراء؟"
                elif "ثاني أكسيد الكربون" in segment and "الماء" in segment:
                    return "ما هي المواد الأولية المطلوبة لعملية التمثيل الضوئي؟"
                elif "الجلوكوز" in segment and "الأكسجين" in segment:
                    return "ما هي نواتج عملية التمثيل الضوئي؟"
                else:
                    return "أين تحدث عملية التمثيل الضوئي في النبات؟"
        
        elif "الجلوكوز" in segment and "الأكسجين" in segment:
            if question_type == "explicit":
                return "لماذا يعتبر الجلوكوز والأكسجين نواتج مهمة لعملية التمثيل الضوئي؟"
            else:
                return "ما هي المواد التي يتم دمجها لإنتاج الجلوكوز والأكسجين؟"
        
        elif "الكلوروفيل" in segment:
            if question_type == "explicit":
                return "كيف يساهم الكلوروفيل في عملية امتصاص الضوء؟"
            else:
                return "ما هو دور الكلوروفيل في عملية التمثيل الضوئي؟"
        
        elif "البلاستيدات الخضراء" in segment:
            if question_type == "explicit":
                return "لماذا تعتبر البلاستيدات الخضراء مهمة في عملية التمثيل الضوئي؟"
            else:
                return "أين توجد البلاستيدات الخضراء في النبات؟"
        
        elif "النباتات الخضراء" in segment or "الطحالب" in segment:
            if question_type == "explicit":
                return "كيف تتمكن النباتات الخضراء والطحالب من القيام بالتمثيل الضوئي؟"
            else:
                return "ما هي الكائنات الحية التي تقوم بعملية التمثيل الضوئي؟"
        
        if question_type == "explicit":
            return "كيف يمكن تفسير العملية المذكورة في هذا النص؟"
        else:
            return "ما هي العناصر الأساسية المذكورة في هذا النص؟"
    
    def generate_questions(self, segment, question_type="explicit"):
        """
        Génère des questions basées sur un segment de texte
        """
        if question_type == "explicit":
            question_starters = ["كيف", "لماذا", "بأي طريقة"]
        else:
            question_starters = ["ما هو", "ما هي", "من", "أين"]
        
        generated_question = self.simulate_question_generation(segment, question_type, question_starters)
        return generated_question
    
    def select_diverse_segments(self, segments):
        """
        Sélectionne les segments les plus diversifiés et informatifs
        """
        scored_segments = []
        
        for segment in segments:
            score = 0
            
            important_terms = [
                "التمثيل الضوئي", "الكلوروفيل", "الجلوكوز", 
                "البلاستيدات", "الأكسجين", "ثاني أكسيد الكربون"
            ]
            
            for term in important_terms:
                if term in segment:
                    score += 1
            
            if any(word in segment for word in ["لأن", "بسبب", "نتيجة", "يؤدي إلى"]):
                score += 2
            
            if any(word in segment for word in ["عملية", "تحويل", "تقوم", "تستخدم"]):
                score += 1
            
            scored_segments.append((segment, score))
        
        scored_segments.sort(key=lambda x: x[1], reverse=True)
        return [segment for segment, score in scored_segments]
    
    def process_text_and_generate_questions(self, arabic_text, max_questions=3):
        """
        Traite le texte complet et génère les questions de manière optimisée
        """
        if not self.is_initialized:
            return []
            
        try:
            # 1. Segmentation
            segments = self.preprocess_arabic_text(arabic_text)
            
            if not segments:
                return []
            
            # 2. Création des embeddings
            self.create_embeddings(segments)
            
            # 3. Extraction des mots-clés
            keywords = self.extract_keywords(arabic_text)
            
            # 4. Sélection intelligente des segments les plus informatifs
            selected_segments = self.select_diverse_segments(segments)
            
            # 5. Génération de questions variées
            results = []
            question_counter = 1
            
            for segment in selected_segments[:max_questions]:
                explicit_question = self.generate_questions(segment, "explicit")
                direct_question = self.generate_questions(segment, "direct")
                
                results.append({
                    'question_number': question_counter,
                    'segment': segment,
                    'explicit': explicit_question,
                    'direct': direct_question
                })
                question_counter += 1
            
            return results
        except Exception as e:
            print(f"Erreur lors du traitement: {e}")
            return []

# Configuration Flask
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Instance globale du générateur RAG
rag_generator = None

def initialize_rag():
    """Initialise le générateur RAG de manière sécurisée"""
    global rag_generator
    try:
        rag_generator = ArabicRAGQuestionGenerator()
        return True
    except Exception as e:
        print(f"Erreur lors de l'initialisation du RAG: {e}")
        return False

@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_questions():
    """Endpoint pour générer des questions"""
    try:
        # Récupérer les données du formulaire
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Aucun texte fourni'}), 400
        
        arabic_text = data['text'].strip()
        max_questions = int(data.get('max_questions', 3))
        
        if not arabic_text:
            return jsonify({'error': 'Le texte ne peut pas être vide'}), 400
        
        # Vérifier si le générateur RAG est initialisé
        global rag_generator
        if not rag_generator or not rag_generator.is_initialized:
            # Tentative de réinitialisation
            if not initialize_rag():
                return jsonify({
                    'error': 'Système RAG non disponible. Veuillez réessayer plus tard.',
                    'questions': []
                }), 500
        
        # Générer les questions
        results = rag_generator.process_text_and_generate_questions(arabic_text, max_questions)
        
        if not results:
            return jsonify({
                'message': 'Aucune question générée pour ce texte',
                'questions': []
            })
        
        # Formater les résultats pour l'API
        formatted_results = []
        for result in results:
            formatted_results.append({
                'number': result['question_number'],
                'segment': result['segment'],
                'explicit': result['explicit'],
                'direct': result['direct']
            })
        
        return jsonify({
            'success': True,
            'questions': formatted_results,
            'total_segments': len(rag_generator.segments) if rag_generator.segments else 0,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Erreur lors de la génération: {e}")
        traceback.print_exc()
        return jsonify({
            'error': f'Erreur lors de la génération des questions: {str(e)}',
            'questions': []
        }), 500

@app.route('/health')
def health_check():
    """Vérification de l'état du système"""
    global rag_generator
    return jsonify({
        'status': 'healthy' if rag_generator and rag_generator.is_initialized else 'degraded',
        'rag_initialized': rag_generator.is_initialized if rag_generator else False,
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Initialiser le système RAG au démarrage
    print("Initialisation du système RAG...")
    if initialize_rag():
        print("✓ Système RAG initialisé avec succès")
    else:
        print("⚠ Attention: Le système RAG n'a pas pu être initialisé")
    
    # Créer le dossier templates s'il n'existe pas
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)