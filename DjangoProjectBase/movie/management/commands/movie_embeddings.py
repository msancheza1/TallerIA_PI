import os
import numpy as np
import random
from django.core.management.base import BaseCommand
from movie.models import Movie
from openai import OpenAI
from dotenv import load_dotenv

class Command(BaseCommand):
    help = "Generate and store embedding for ONE random movie"

    def handle(self, *args, **kwargs):
        load_dotenv()

        api_key = os.environ.get('openai_apikey')
        use_api = True

        # 🔹 Intentar usar OpenAI
        try:
            client = OpenAI(api_key=api_key)
            if not api_key:
                raise Exception("No API key found")
        except Exception:
            self.stdout.write(".")
            use_api = False

        movies = list(Movie.objects.all())

        if not movies:
            self.stdout.write("No movies found.")
            return

        self.stdout.write(f"Found {len(movies)} movies in the database")

        # 🔹 Función API
        def get_embedding_api(text):
            response = client.embeddings.create(
                input=[text],
                model="text-embedding-3-small"
            )
            return np.array(response.data[0].embedding, dtype=np.float32)

        # 🔹 Función manual
        def get_embedding_manual(text):
            np.random.seed(abs(hash(text)) % (10**8))
            return np.random.rand(1536).astype(np.float32)

        # 🎯 SOLO UNA película aleatoria
        movie = random.choice(movies)

        try:
            if use_api:
                emb = get_embedding_api(movie.description)
            else:
                emb = get_embedding_manual(movie.description)

            movie.emb = emb.tobytes()
            movie.save()

            self.stdout.write(self.style.SUCCESS(f"✅ Embedding stored for: {movie.title}"))

        except Exception:
            self.stderr.write(f"❌ Error con API en {movie.title}, usando fallback manual")

            emb = get_embedding_manual(movie.description)
            movie.emb = emb.tobytes()
            movie.save()

            self.stdout.write(self.style.SUCCESS(f"⚡ Manual embedding stored for: {movie.title}"))

        # 👀 Mostrar parte del embedding (para captura)
        vector = np.frombuffer(movie.emb, dtype=np.float32)
        self.stdout.write("🔢 Primeros 10 valores del embedding:")
        self.stdout.write(str(vector[:10]))

        self.stdout.write(self.style.SUCCESS("🎯 Finished generating embedding for one movie"))