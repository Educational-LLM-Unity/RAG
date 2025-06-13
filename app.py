from flask import Flask, render_template, request, jsonify, flash
import os
import json
from datetime import datetime
import traceback
import warnings
warnings.filterwarnings("ignore", message=".*sparse_softmax_cross_entropy.*")
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM



# Import du système RAG générique
from generic_arabic_rag import ImprovedArabicRAGQuestionGenerator

# Configuration Flask
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Instance globale du générateur RAG
rag_generator = None

def initialize_rag():
    """Initialise le générateur RAG de manière sécurisée"""
    global rag_generator
    try:
        rag_generator = ImprovedArabicRAGQuestionGenerator()
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
        
        # Validation minimale du texte arabe
        if len(arabic_text) < 20:
            return jsonify({'error': 'Le texte est trop court pour générer des questions pertinentes'}), 400
        
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
                'message': 'Aucune question générée pour ce texte. Veuillez vérifier que le texte contient suffisamment d\'informations.',
                'questions': []
            })
        
        # Formater les résultats pour l'API
        formatted_results = []
        for result in results:
            formatted_results.append({
                'number': result['question_number'],
                'segment': result['segment'],
                'explicit': result['explicit'],
                'direct': result['direct'],
                'domain': result.get('domain', 'غير محدد')
            })
        
        return jsonify({
            'success': True,
            'questions': formatted_results,
            'total_segments': len(rag_generator.segments) if rag_generator.segments else 0,
            'detected_domain': results[0].get('domain', 'غير محدد') if results else 'غير محدد',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Erreur lors de la génération: {e}")
        traceback.print_exc()
        return jsonify({
            'error': f'Erreur lors de la génération des questions: {str(e)}',
            'questions': []
        }), 500

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """Endpoint pour analyser le texte et retourner des informations"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Aucun texte fourni'}), 400
        
        arabic_text = data['text'].strip()
        
        if not arabic_text:
            return jsonify({'error': 'Le texte ne peut pas être vide'}), 400
        
        global rag_generator
        if not rag_generator or not rag_generator.is_initialized:
            if not initialize_rag():
                return jsonify({'error': 'Système RAG non disponible'}), 500
        
        # Analyser le texte
        domain = rag_generator.detect_text_domain(arabic_text)
        entities = rag_generator.extract_key_entities(arabic_text)
        segments = rag_generator.preprocess_arabic_text(arabic_text)
        
        return jsonify({
            'success': True,
            'domain': domain,
            'entities': entities,
            'segments_count': len(segments),
            'text_length': len(arabic_text),
            'analysis': {
                'domain_detected': domain,
                'has_entities': any(len(entity_list) > 0 for entity_list in entities.values()),
                'segments_available': len(segments) > 0,
                'suitable_for_questions': len(segments) > 0 and len(arabic_text) > 20
            }
        })
        
    except Exception as e:
        print(f"Erreur lors de l'analyse: {e}")
        return jsonify({'error': f'Erreur lors de l\'analyse: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Vérification de l'état du système"""
    global rag_generator
    return jsonify({
        'status': 'healthy' if rag_generator and rag_generator.is_initialized else 'degraded',
        'rag_initialized': rag_generator.is_initialized if rag_generator else False,
        'domains_supported': list(rag_generator.domain_patterns.keys()) if rag_generator else [],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/domains')
def get_supported_domains():
    """Retourne les domaines supportés"""
    global rag_generator
    if rag_generator:
        return jsonify({
            'domains': {
                'science': 'علمي',
                'history': 'تاريخي', 
                'literature': 'أدبي',
                'geography': 'جغرافي',
                'religion': 'ديني',
                'general': 'عام'
            },
            'patterns': rag_generator.domain_patterns
        })
    else:
        return jsonify({'error': 'Système non initialisé'}), 500

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Initialiser le système RAG au démarrage
    print("Initialisation du système RAG générique...")
    if initialize_rag():
        print("✓ Système RAG générique initialisé avec succès")
        print("✓ Domaines supportés:", list(rag_generator.domain_patterns.keys()))
    else:
        print("⚠ Attention: Le système RAG n'a pas pu être initialisé")
    
    # Créer les dossiers nécessaires
    os.makedirs('templates', exist_ok=True)
    #os.makedirs('static/css', exist_ok=True)
    #os.makedirs('static/js', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)