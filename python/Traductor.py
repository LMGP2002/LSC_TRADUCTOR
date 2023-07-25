from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import json

# Descarga el modelo pre-entrenado en español
model = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')

app = Flask(__name__)

# Cargar los datos del JSON con codificación UTF-8
with open('señas.json', encoding='utf-8') as json_file:
    data = json.load(json_file)

def comparar_oraciones(oracion1, oracion2):
    # Generar los embeddings de las oraciones
    embedding1 = model.encode(oracion1, convert_to_tensor=True)
    embedding2 = model.encode(oracion2, convert_to_tensor=True)

    # Calcular la similitud coseno entre los embeddings
    similitud = util.pytorch_cos_sim(embedding1, embedding2)[0][0]

    # Convertir la similitud a un valor entre 0 y 1
    similitud = similitud.item()

    return similitud

@app.route('/analizar', methods=['GET'])
def analizar_texto():
    texto = request.args.get('text')
    resultados = []
    for clave, valor in data.items():
        similitud = comparar_oraciones(clave, texto)
        resultados.append({'valor': valor, 'similitud': similitud})
    resultados = sorted(resultados, key=lambda x: x['similitud'], reverse=True)

    similitud_maxima = round(resultados[0]['similitud'],2)
    if similitud_maxima  <0.60:
        mejor_coincidencia="No existen coincidencias"
    else: 
        mejor_coincidencia = resultados[0]['valor']
        
    response = f"{mejor_coincidencia}_{similitud_maxima}" 
    return response

if __name__ == '__main__':
    app.run()
