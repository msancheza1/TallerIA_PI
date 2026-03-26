import os
import numpy as np
from django.core.management.base import BaseCommand
from movie.models import Movie
from openai import OpenAI
from dotenv import load_dotenv


class Command(BaseCommand):
    help = "Compare two movies using embeddings (IA + fallback)"

    def handle(self, *args, **kwargs):
        load_dotenv()

        client = None

        # 🔥 Intentar usar OpenAI correctamente
        try:
            api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("openai_apikey")

            if api_key:
                client = OpenAI(api_key=api_key)
            else:
                print(".")

        except Exception as e:
            print("⚠️ No se pudo inicializar OpenAI:", e)

        # 🔥 IA
        def get_embedding_openai(text):
            response = client.embeddings.create(
                input=[text],
                model="text-embedding-3-small"
            )
            return np.array(response.data[0].embedding, dtype=np.float32)

        # 🔥 FALLBACK (sin IA)
        def get_embedding_local(text):
            if not text:
                return np.zeros(50)

            words = text.lower().split()
            vector = np.zeros(50)

            for word in words:
                index = hash(word) % 50
                vector[index] += 1

            return vector

        # 🔥 FUNCIÓN GENERAL
        def get_embedding(text):
            if client:
                try:
                    return get_embedding_openai(text)
                except Exception as e:
                    print("⚠️ Falló IA, usando fallback:", e)
                    return get_embedding_local(text)
            else:
                return get_embedding_local(text)

        # 🔥 SIMILITUD
        def cosine_similarity(a, b):
            if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
                return 0
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        # ✅ Obtener películas de forma segura (NO usar get directo)
        movies = Movie.objects.all()

        if len(movies) < 2:
            self.stdout.write("❌ No hay suficientes películas en la BD")
            return

        movie1 = movies[0]
        movie2 = movies[1]

        # 🔥 evitar None
        desc1 = movie1.description or ""
        desc2 = movie2.description or ""

        # 🔥 embeddings
        emb1 = get_embedding(desc1)
        emb2 = get_embedding(desc2)

        similarity = cosine_similarity(emb1, emb2)

        self.stdout.write(
            f"🎬 Similaridad entre '{movie1.title}' y '{movie2.title}': {similarity:.4f}"
        )

        # 🔥 prompt
        prompt = "película sobre la Segunda Guerra Mundial"
        prompt_emb = get_embedding(prompt)

        sim1 = cosine_similarity(prompt_emb, emb1)
        sim2 = cosine_similarity(prompt_emb, emb2)

        self.stdout.write(f"📝 Prompt vs '{movie1.title}': {sim1:.4f}")
        self.stdout.write(f"📝 Prompt vs '{movie2.title}': {sim2:.4f}")