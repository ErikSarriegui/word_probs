from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import gradio as gr

def cargar_modelo_y_tokenizador(nombre_modelo):
    """Carga y devuelve el tokenizador y el modelo preentrenado"""
    tokenizer = AutoTokenizer.from_pretrained(nombre_modelo)
    model = AutoModelForMaskedLM.from_pretrained(nombre_modelo)
    return tokenizer, model

def encontrar_mascaras(inputs):
    """Identifica las posiciones de los tokens [MASK] en los inputs"""
    indices = torch.where(inputs["input_ids"] == 103)  # 103 es el ID de [MASK]
    return list(zip(indices[0].tolist(), indices[1].tolist()))

def procesar_frases(frases):
    """Extrae textos y palabras candidatas de la lista de frases"""
    textos = [frase["text"] for frase in frases]
    palabras_candidatas = [frase["words"] for frase in frases]
    return textos, palabras_candidatas

def tokenizar_textos(tokenizer, textos):
    """Tokeniza múltiples textos con padding"""
    return tokenizer(textos, return_tensors="pt", padding=True)

def obtener_logits(modelo, inputs):
    """Ejecuta el modelo y devuelve los logits con softmax aplicado"""
    outputs = modelo(**inputs)
    return outputs.logits.softmax(dim=-1)

def evaluar_candidatos(logits, posiciones_mascaras, palabras_candidatas, tokenizer):
    """Evalúa las palabras candidatas y devuelve todas las opciones con sus probabilidades en el formato solicitado"""
    resultados = []
    
    for i, (batch_idx, pos) in enumerate(posiciones_mascaras):
        candidatos = palabras_candidatas[i]
        candidatos_con_probs = []
        
        for palabra in candidatos:
            tokenizado = tokenizer.tokenize(palabra)
            if not tokenizado:
                continue
            
            token_id = tokenizer.convert_tokens_to_ids(tokenizado)
            prob = logits[batch_idx, pos, token_id[0]].item()
            candidatos_con_probs.append((palabra, prob))
        
        # Ordenar por probabilidad descendente
        candidatos_con_probs.sort(reverse=True, key=lambda x: x[1])
        
        # Convertir a diccionario de palabras:probabilidades
        palabras_probs = {palabra: round(prob, 4) for palabra, prob in candidatos_con_probs}
        
        # Agregar al resultado final
        resultados.append({"words": palabras_probs})
    
    return resultados

def evaluate_word_probs(frases, model_id):
    tokenizer, model = cargar_modelo_y_tokenizador(model_id)

    textos, palabras_candidatas = procesar_frases(frases)
    inputs = tokenizar_textos(tokenizer, textos)
    posiciones_mascaras = encontrar_mascaras(inputs)

    logits = obtener_logits(model, inputs)

    resultados = evaluar_candidatos(
        logits, posiciones_mascaras, palabras_candidatas, tokenizer
    )
    return resultados

def predict(text, candidates, model_id):
    candidates_list = [c.strip() for c in candidates.split(';') if c.strip()]
    frases = [{"text": text, "words": candidates_list}]
    results = evaluate_word_probs(frases, model_id)
    
    output = []
    for result in results:
        for word, prob in result['words'].items():
            output.append([word, str(prob*100) + "%"])
    
    return output

iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Texto con [MASK]", placeholder="Ej: [MASK] is a fashion model"),
        gr.Textbox(label="Candidatos (separados por ;)", placeholder="Ej: He; She"),
        gr.Textbox(label="Model ID (Debe ser un encoder)", value="google-bert/bert-base-uncased")
    ],

    outputs = gr.Dataframe(
        headers=["Candidato", "Probabilidad"],
        datatype=["str", "str"],
        label="Resultados de Predicción"
    ),

    title="Comparación de probabilidades de palabras",
    description="""Este espacio recibe como entrada una frase y varias palabras con el objetivo de ver qué palabra cree el modelo del lenguaje que es más probable.
    Cabe mencionar que el output del modelo, en porcentaje, no suma 100% ya que Softmax está aplicado a todos los posibles tokens, es decir, qué quiere decir que la probabilidad no es frente a la otra palabra, sino al globar de todo el vocabulario.
    """
)

iface.launch(server_name="0.0.0.0", server_port=7860)