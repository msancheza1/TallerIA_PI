import numpy as np
import random
from django.core.management.base import BaseCommand
from movie.models import Movie

class Command(BaseCommand):
    help = "Show embedding of a random movie"

    def handle(self, *args, **kwargs):
        movies = Movie.objects.all()

        if not movies.exists():
            self.stdout.write("No movies found.")
            return

        movie = random.choice(movies)

        embedding_vector = np.frombuffer(movie.emb, dtype=np.float32)

        self.stdout.write(f"🎬 Movie: {movie.title}")
        self.stdout.write(f"🔢 First 10 values of embedding:")
        self.stdout.write(str(embedding_vector[:10]))