"""Tests para las estructuras de datos base (Triangle, Individual)."""

import pytest
from src.genetic.individual import Triangle, Individual


class TestTriangle:
    """Tests para la clase Triangle."""

    def test_create_valid_triangle(self):
        """Debe crear un triángulo con valores válidos."""
        vertices = [(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)]
        color = (255, 128, 0, 0.5)

        triangle = Triangle(vertices=vertices, color=color)

        assert triangle.vertices == vertices
        assert triangle.color == color

    def test_invalid_vertex_count(self):
        """Debe fallar si no hay exactamente 3 vértices."""
        with pytest.raises(ValueError, match="exactamente 3 vértices"):
            Triangle(vertices=[(0, 0), (1, 1)], color=(0, 0, 0, 0.5))

    def test_invalid_vertex_range(self):
        """Debe fallar si las coordenadas están fuera de [0, 1]."""
        with pytest.raises(ValueError, match="fuera de rango"):
            Triangle(vertices=[(0, 0), (1.5, 0.5), (0.5, 0.5)], color=(0, 0, 0, 0.5))

    def test_invalid_color_rgb(self):
        """Debe fallar si RGB está fuera de [0, 255]."""
        with pytest.raises(ValueError, match="RGB fuera de rango"):
            Triangle(vertices=[(0, 0), (1, 0), (0.5, 1)], color=(256, 0, 0, 0.5))

    def test_invalid_alpha(self):
        """Debe fallar si alfa está fuera de [0, 1]."""
        with pytest.raises(ValueError, match="alfa fuera de rango"):
            Triangle(vertices=[(0, 0), (1, 0), (0.5, 1)], color=(0, 0, 0, 1.5))

    def test_random_triangle(self):
        """Debe generar un triángulo aleatorio válido."""
        triangle = Triangle.random(alpha_min=0.2, alpha_max=0.7)

        assert len(triangle.vertices) == 3
        for x, y in triangle.vertices:
            assert 0 <= x <= 1
            assert 0 <= y <= 1

        r, g, b, a = triangle.color
        assert 0 <= r <= 255
        assert 0 <= g <= 255
        assert 0 <= b <= 255
        assert 0.2 <= a <= 0.7

    def test_to_absolute(self):
        """Debe convertir coordenadas normalizadas a absolutas."""
        triangle = Triangle(
            vertices=[(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)], color=(100, 150, 200, 0.5)
        )

        abs_vertices, abs_color = triangle.to_absolute(width=100, height=200)

        assert abs_vertices == [(0, 0), (50, 100), (100, 200)]
        assert abs_color == (100, 150, 200, 127)  # 0.5 * 255 ≈ 127

    def test_copy(self):
        """Debe crear una copia independiente."""
        original = Triangle.random()
        copied = original.copy()

        assert original.vertices == copied.vertices
        assert original.color == copied.color
        assert original is not copied

    def test_serialization(self):
        """Debe serializar y deserializar correctamente."""
        original = Triangle(
            vertices=[(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)], color=(10, 20, 30, 0.4)
        )

        data = original.to_dict()
        restored = Triangle.from_dict(data)

        assert original.vertices == restored.vertices
        assert original.color == restored.color


class TestIndividual:
    """Tests para la clase Individual."""

    def test_create_individual(self):
        """Debe crear un individuo con triángulos."""
        triangles = [Triangle.random() for _ in range(5)]
        individual = Individual(triangles=triangles)

        assert len(individual) == 5
        assert individual.fitness is None

    def test_random_individual(self):
        """Debe generar un individuo aleatorio."""
        individual = Individual.random(num_triangles=10)

        assert len(individual) == 10
        assert all(isinstance(t, Triangle) for t in individual.triangles)

    def test_indexing(self):
        """Debe permitir acceso por índice."""
        individual = Individual.random(num_triangles=5)

        triangle = individual[0]
        assert isinstance(triangle, Triangle)

        new_triangle = Triangle.random()
        individual[0] = new_triangle
        assert individual[0] is new_triangle

    def test_fitness_invalidation(self):
        """Debe invalidar fitness al modificar triángulos."""
        individual = Individual.random(num_triangles=5)
        individual.fitness = 0.5

        individual[0] = Triangle.random()

        assert individual.fitness is None

    def test_copy(self):
        """Debe crear una copia profunda independiente."""
        original = Individual.random(num_triangles=5)
        original.fitness = 0.123
        copied = original.copy()

        assert len(original) == len(copied)
        assert original.fitness == copied.fitness
        assert original is not copied
        assert original.triangles[0] is not copied.triangles[0]

    def test_serialization(self):
        """Debe serializar y deserializar correctamente."""
        original = Individual.random(num_triangles=3)
        original.fitness = 0.456

        data = original.to_dict()
        restored = Individual.from_dict(data)

        assert len(original) == len(restored)
        assert original.fitness == restored.fitness
        for orig_t, rest_t in zip(original.triangles, restored.triangles):
            assert orig_t.vertices == rest_t.vertices
            assert orig_t.color == rest_t.color
