import os
import random
from openai import OpenAI
from django.core.management.base import BaseCommand
from movie.models import Movie
from dotenv import load_dotenv

class Command(BaseCommand):
    help = "Update movie descriptions using AI with fallback"

    def generate_manual_description(self, movie):
        title = movie.title.lower()

        if any(word in title for word in ["love", "amor"]):
            genre = "romance"
        elif any(word in title for word in ["war", "battle"]):
            genre = "acción"
        elif any(word in title for word in ["ghost", "horror"]):
            genre = "terror"
        else:
            genre = "drama"

        templates = [
            f"{movie.title} es una película de {genre} con una historia envolvente y personajes interesantes.",
            f"Esta película pertenece al género {genre} y presenta una narrativa atractiva ideal para los fans del género.",
            f"{movie.title} ofrece una experiencia del género {genre}, combinando emoción y desarrollo de personajes."
        ]

        return random.choice(templates)

    def handle(self, *args, **kwargs):
        load_dotenv()

        client = None

        # 🔥 Intentar inicializar IA
        try:
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        except Exception as e:
            print("⚠️ No se pudo inicializar OpenAI:", e)

        def get_ai_description(prompt):
            try:
                response = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print("⚠️ Error con IA:", e)
                return None

        instruction = (
            "Describe la película en menos de 200 palabras, incluyendo género "
            "y detalles útiles para recomendación."
        )

        movies = Movie.objects.all()
        self.stdout.write(f"Found {movies.count()} movies")

        for movie in movies:
            self.stdout.write(f"Processing: {movie.title}")

            try:
                prompt = f"{instruction} Película: {movie.title}. Descripción actual: {movie.description}"

                updated_description = None

                # 🔥 Intentar IA si existe
                if client:
                    updated_description = get_ai_description(prompt)

                # 🔥 FALLBACK automático
                if not updated_description:
                    print("👉 Usando método manual")
                    updated_description = self.generate_manual_description(movie)

                movie.description = updated_description
                movie.save()

                self.stdout.write(self.style.SUCCESS(f"Updated: {movie.title}"))

            except Exception as e:
                self.stderr.write(f"Failed for {movie.title}: {str(e)}")

            break  