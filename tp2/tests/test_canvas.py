"""Tests para el motor de renderizado (Canvas)."""

import pytest
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile

from src.genetic.individual import Triangle, Individual
from src.rendering.canvas import Canvas, load_target_image, resize_image


class TestCanvas:
    """Tests para la clase Canvas."""

    def test_create_canvas(self):
        """Debe crear un lienzo con dimensiones válidas."""
        canvas = Canvas(width=100, height=200)

        assert canvas.width == 100
        assert canvas.height == 200

    def test_invalid_dimensions(self):
        """Debe fallar con dimensiones inválidas."""
        with pytest.raises(ValueError):
            Canvas(width=0, height=100)

        with pytest.raises(ValueError):
            Canvas(width=100, height=-1)

    def test_from_image(self):
        """Debe crear canvas con dimensiones de una imagen."""
        img = Image.new("RGB", (150, 250))
        canvas = Canvas.from_image(img)

        assert canvas.width == 150
        assert canvas.height == 250

    def test_render_empty_individual(self):
        """Debe renderizar un individuo sin triángulos como imagen blanca."""
        canvas = Canvas(width=50, height=50)
        individual = Individual(triangles=[])

        image = canvas.render(individual)

        assert image.mode == "RGB"
        assert image.size == (50, 50)

        # Verificar que es completamente blanca
        arr = np.array(image)
        assert np.all(arr == 255)

    def test_render_single_triangle(self):
        """Debe renderizar un triángulo visible."""
        canvas = Canvas(width=100, height=100)

        # Triángulo rojo opaco que cubre parte del lienzo
        triangle = Triangle(
            vertices=[(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)],
            color=(255, 0, 0, 1.0),  # Rojo completamente opaco
        )
        individual = Individual(triangles=[triangle])

        image = canvas.render(individual)
        arr = np.array(image)

        # El centro del triángulo debería ser rojo
        center_pixel = arr[50, 50]
        assert center_pixel[0] == 255  # R
        assert center_pixel[1] == 0  # G
        assert center_pixel[2] == 0  # B

    def test_render_transparency(self):
        """Debe mezclar colores con transparencia correctamente."""
        canvas = Canvas(width=100, height=100)

        # Triángulo azul semi-transparente
        triangle = Triangle(
            vertices=[(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)],
            color=(0, 0, 255, 0.5),  # Azul 50% transparente
        )
        individual = Individual(triangles=[triangle])

        image = canvas.render(individual)
        arr = np.array(image)

        # El centro debería ser una mezcla de azul y blanco
        center_pixel = arr[50, 50]
        # Con alpha 0.5: resultado ≈ 0.5 * azul + 0.5 * blanco
        # El canal azul debería ser mayor que rojo y verde
        assert center_pixel[2] >= center_pixel[0]  # Más o igual azul que rojo
        assert center_pixel[2] >= center_pixel[1]  # Más o igual azul que verde

    def test_render_z_order(self):
        """Debe respetar el orden Z (último triángulo encima)."""
        canvas = Canvas(width=100, height=100)

        # Dos triángulos superpuestos, verde abajo, rojo arriba
        green = Triangle(
            vertices=[(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)], color=(0, 255, 0, 1.0)
        )
        red = Triangle(
            vertices=[(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)], color=(255, 0, 0, 1.0)
        )
        individual = Individual(triangles=[green, red])  # Rojo encima

        image = canvas.render(individual)
        arr = np.array(image)

        # El centro debería ser rojo (último renderizado)
        center_pixel = arr[50, 50]
        assert center_pixel[0] == 255  # R
        assert center_pixel[1] == 0  # G

    def test_render_to_array(self):
        """Debe retornar array NumPy con forma correcta."""
        canvas = Canvas(width=80, height=60)
        individual = Individual.random(num_triangles=5)

        arr = canvas.render_to_array(individual)

        assert isinstance(arr, np.ndarray)
        assert arr.shape == (60, 80, 3)  # (height, width, channels)
        assert arr.dtype == np.uint8

    def test_save_image(self):
        """Debe guardar la imagen renderizada."""
        canvas = Canvas(width=50, height=50)
        individual = Individual.random(num_triangles=3)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name

        try:
            canvas.save(individual, path)

            # Verificar que el archivo existe y es válido
            assert Path(path).exists()
            loaded = Image.open(path)
            assert loaded.size == (50, 50)
        finally:
            Path(path).unlink(missing_ok=True)


class TestImageUtils:
    """Tests para funciones de utilidad de imágenes."""

    def test_load_target_image(self):
        """Debe cargar imagen y convertir a array."""
        # Crear imagen temporal de prueba
        img = Image.new("RGB", (30, 40), color=(100, 150, 200))

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
            img.save(path)

        try:
            loaded_img, arr = load_target_image(path)

            assert loaded_img.size == (30, 40)
            assert arr.shape == (40, 30, 3)
            assert arr.dtype == np.uint8
            # Verificar color
            assert np.all(arr[:, :, 0] == 100)
            assert np.all(arr[:, :, 1] == 150)
            assert np.all(arr[:, :, 2] == 200)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_resize_image_larger(self):
        """Debe redimensionar imagen grande."""
        img = Image.new("RGB", (500, 300))

        resized = resize_image(img, max_size=100)

        assert resized.width == 100
        assert resized.height == 60  # Mantiene relación de aspecto

    def test_resize_image_smaller(self):
        """No debe modificar imagen ya pequeña."""
        img = Image.new("RGB", (50, 30))

        resized = resize_image(img, max_size=100)

        assert resized.width == 50
        assert resized.height == 30

    def test_resize_preserves_aspect_ratio(self):
        """Debe mantener la relación de aspecto."""
        img = Image.new("RGB", (400, 200))  # 2:1

        resized = resize_image(img, max_size=100)

        ratio_original = 400 / 200
        ratio_resized = resized.width / resized.height
        assert abs(ratio_original - ratio_resized) < 0.01
