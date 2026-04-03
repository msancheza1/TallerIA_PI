from django.shortcuts import render
from openai import OpenAI
import numpy as np
import os
from dotenv import load_dotenv
from movie.models import Movie

load_dotenv('openAI.env')
client = OpenAI(api_key=os.environ.get('openai_apikey'))


def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)


def recommender_view(request):
    context = {
        'movie': None,
        'similarity': None,
        'prompt': '',
        'error': None,
    }

    if request.method == 'POST':
        prompt = request.POST.get('prompt', '').strip()
        context['prompt'] = prompt

        if not prompt:
            context['error'] = 'Por favor ingresa una descripción.'
            return render(request, 'recommender/recommender.html', context)

        try:
            # 1. Generar embedding del prompt
            response = client.embeddings.create(
                input=[prompt],
                model="text-embedding-3-small"
            )
            prompt_emb = np.array(response.data[0].embedding, dtype=np.float32)

            # 2. Comparar con cada película
            best_movie = None
            max_sim = -1

            for movie in Movie.objects.all():
                if movie.emb:
                    # ✅ bytes() es clave — Django devuelve memoryview
                    movie_emb = np.frombuffer(bytes(movie.emb), dtype=np.float32)
                    if movie_emb.shape[0] == 0:
                        continue
                    sim = cosine_similarity(prompt_emb, movie_emb)
                    if sim > max_sim:
                        max_sim = sim
                        best_movie = movie

            if best_movie:
                context['movie'] = best_movie
                context['similarity'] = round(float(max_sim) * 100, 2)
            else:
                context['error'] = 'Sin resultados. ¿Están generados los embeddings?'

        except Exception as e:
            context['error'] = f'Error: {str(e)}'

    return render(request, 'recommender/recommender.html', context)

    context['similarity'] = round(abs(float(max_sim)) * 100, 2)