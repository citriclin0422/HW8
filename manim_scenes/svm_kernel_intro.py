"""Manim concept scene for SVM, margin, support vectors, and kernel trick."""

from __future__ import annotations

import numpy as np
from manim import (
    BLUE,
    DEGREES,
    GREEN,
    GREY_B,
    GREY_D,
    ORANGE,
    PURPLE,
    RED,
    WHITE,
    YELLOW,
    Arrow,
    Circle,
    DashedLine,
    Dot,
    FadeIn,
    FadeOut,
    GrowArrow,
    Line,
    Rectangle,
    Scene,
    Surface,
    Text,
    ThreeDAxes,
    ThreeDScene,
    Transform,
    VGroup,
    Write,
    Create,
)


class SVMKernelIntro(ThreeDScene):
    """A longer teaching animation for SVM and the kernel trick."""

    def construct(self):
        self._opening()
        self._linear_margin_scene()
        self._nonlinear_problem_scene()
        self._kernel_mapping_scene()
        self._rbf_summary_scene()

    def _opening(self):
        title = Text("Support Vector Machine", font_size=44, color=BLUE).to_edge(np.array([0, 1, 0]))
        subtitle = Text("Maximum margin -> support vectors -> kernel trick", font_size=26).next_to(title, np.array([0, -1, 0]))
        roadmap = VGroup(
            Text("1. Find the widest margin", font_size=24),
            Text("2. Keep the critical points: support vectors", font_size=24),
            Text("3. Map nonlinear data into a feature space", font_size=24),
            Text("4. Separate with a plane, project back to a curve", font_size=24),
        ).arrange(np.array([0, -1, 0]), aligned_edge=np.array([-1, 0, 0]), buff=0.22).shift((-1.8, -0.6, 0))

        self.play(Write(title), FadeIn(subtitle))
        self.play(FadeIn(roadmap, shift=np.array([0, -0.2, 0])))
        self.wait(2)
        self.play(FadeOut(title), FadeOut(subtitle), FadeOut(roadmap))

    def _linear_margin_scene(self):
        section = Text("Linear SVM: choose the widest safe street", font_size=34, color=BLUE).to_edge(np.array([0, 1, 0]))
        axes = self._make_2d_axes()
        blue_points = np.array([[-2.6, 1.1], [-2.1, 1.6], [-1.8, 0.6], [-1.5, 1.2], [-1.0, 1.7], [-0.8, 0.8]])
        orange_points = np.array([[0.8, -0.8], [1.1, -1.6], [1.5, -0.4], [1.8, -1.0], [2.2, -0.2], [2.6, -1.3]])
        dots = VGroup(
            *[Dot((x, y, 0), color=BLUE, radius=0.075) for x, y in blue_points],
            *[Dot((x, y, 0), color=ORANGE, radius=0.075) for x, y in orange_points],
        )
        weak_line = Line((-2.8, -1.8, 0), (1.6, 2.0, 0), color=GREY_B)
        strong_line = Line((-1.1, -2.0, 0), (2.3, 1.9, 0), color=YELLOW)
        margin_a = DashedLine((-1.55, -2.0, 0), (1.85, 1.9, 0), color=WHITE)
        margin_b = DashedLine((-0.65, -2.0, 0), (2.75, 1.9, 0), color=WHITE)
        support = VGroup(
            Circle(radius=0.16, color=YELLOW).move_to((-1.0, 1.7, 0)),
            Circle(radius=0.16, color=YELLOW).move_to((0.8, -0.8, 0)),
            Circle(radius=0.16, color=YELLOW).move_to((1.5, -0.4, 0)),
        )
        formula = Text("decision boundary: w^T x + b = 0", font_size=24).to_edge(np.array([0, -1, 0]))
        margin_text = Text("margin width = 2 / ||w||", font_size=24, color=YELLOW).next_to(formula, np.array([0, 1, 0]))
        sv_text = Text("Support vectors are the points touching the margin.", font_size=24, color=YELLOW).to_edge(np.array([0, -1, 0]))

        self.play(Write(section), Create(axes), FadeIn(dots))
        self.play(Create(weak_line), FadeIn(Text("many lines separate the classes", font_size=22).to_edge(np.array([0, -1, 0]))))
        self.wait(1)
        self.play(Transform(weak_line, strong_line), FadeIn(margin_a), FadeIn(margin_b), Transform(formula.copy(), formula))
        self.play(FadeIn(margin_text), FadeIn(support))
        self.wait(2)
        self.play(Transform(formula, sv_text), FadeOut(margin_text))
        self.wait(1.5)
        self.play(FadeOut(section), FadeOut(axes), FadeOut(dots), FadeOut(weak_line), FadeOut(margin_a), FadeOut(margin_b), FadeOut(support), FadeOut(formula))

    def _nonlinear_problem_scene(self):
        section = Text("Nonlinear data: no straight line is enough", font_size=34, color=BLUE).to_edge(np.array([0, 1, 0]))
        axes = self._make_2d_axes()
        points, labels = self._make_ring_points()
        dots = self._dots_2d(points, labels)
        trial_lines = VGroup(
            Line((-3, -1.6, 0), (2.8, 1.5, 0), color=GREY_B),
            Line((-2.6, 1.8, 0), (2.8, -1.2, 0), color=GREY_B),
            Line((-0.2, -2.1, 0), (0.4, 2.1, 0), color=GREY_B),
        )
        note = Text("The center class is surrounded by the outer class.", font_size=24).to_edge(np.array([0, -1, 0]))
        map_note = Text("Idea: lift each point with z = x^2 + y^2", font_size=26, color=YELLOW).to_edge(np.array([0, -1, 0]))

        self.play(Write(section), Create(axes), FadeIn(dots))
        self.play(Create(trial_lines), FadeIn(note))
        self.wait(2)
        self.play(FadeOut(trial_lines), Transform(note, map_note))
        self.wait(1.5)
        self.play(FadeOut(section), FadeOut(axes), FadeOut(dots), FadeOut(note))

    def _kernel_mapping_scene(self):
        title = Text("Kernel view: linear separation in feature space", font_size=32, color=BLUE).to_edge(np.array([0, 1, 0]))
        formula = Text("phi(x, y) = (x, y, x^2 + y^2)", font_size=28, color=YELLOW).to_edge(np.array([0, -1, 0]))
        points, labels = self._make_ring_points()
        axes = ThreeDAxes(
            x_range=(-2.2, 2.2),
            y_range=(-2.2, 2.2),
            z_range=(0, 3.2),
            x_length=5,
            y_length=5,
            z_length=3.2,
        )
        mapped = self._dots_3d(points, labels)
        bowl = Surface(
            lambda u, v: np.array([u, v, u * u + v * v]),
            u_range=(-1.55, 1.55),
            v_range=(-1.55, 1.55),
            resolution=(14, 14),
            fill_opacity=0.25,
            checkerboard_colors=[BLUE, PURPLE],
        )
        plane = Rectangle(width=4.5, height=4.5, color=GREEN, fill_color=GREEN, fill_opacity=0.20)
        plane.rotate(90 * DEGREES, axis=np.array([1, 0, 0])).shift((0, 0, 1.15))
        plane_label = Text("separating plane: z = c", font_size=24, color=GREEN).to_edge(np.array([0, -1, 0]))

        self.play(Write(title), Write(formula))
        self.play(Create(axes), FadeIn(mapped), FadeIn(bowl))
        self.set_camera_orientation(phi=62 * DEGREES, theta=-42 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.18)
        self.wait(2)
        self.play(Create(plane), Transform(formula, plane_label))
        self.wait(4)
        self.stop_ambient_camera_rotation()
        self.play(FadeOut(axes), FadeOut(mapped), FadeOut(bowl), FadeOut(plane), FadeOut(title), FadeOut(formula))

        self.move_camera(phi=0 * DEGREES, theta=-90 * DEGREES)
        projection_title = Text("Projected back to 2D: a curved boundary", font_size=34, color=BLUE).to_edge(np.array([0, 1, 0]))
        axes2 = self._make_2d_axes()
        dots2 = self._dots_2d(points, labels)
        ring_boundary = Circle(radius=1.05, color=YELLOW).set_stroke(width=5)
        projection_note = Text("The plane in 3D becomes a circle in 2D.", font_size=25, color=YELLOW).to_edge(np.array([0, -1, 0]))
        self.play(Write(projection_title), Create(axes2), FadeIn(dots2), Create(ring_boundary), FadeIn(projection_note))
        self.wait(3)
        self.play(FadeOut(projection_title), FadeOut(axes2), FadeOut(dots2), FadeOut(ring_boundary), FadeOut(projection_note))

    def _rbf_summary_scene(self):
        title = Text("Real RBF SVM: no need to draw the full feature space", font_size=32, color=BLUE).to_edge(np.array([0, 1, 0]))
        left = VGroup(
            Text("RBF kernel", font_size=30, color=YELLOW),
            Text("K(x, z) = exp(-gamma * ||x - z||^2)", font_size=24),
            Text("Only similarities are computed.", font_size=24),
        ).arrange(np.array([0, -1, 0]), aligned_edge=np.array([-1, 0, 0]), buff=0.28).shift((-2.6, 0.9, 0))
        right = VGroup(
            Text("Parameter intuition", font_size=30, color=GREEN),
            Text("C small: wider, smoother margin", font_size=23),
            Text("C large: fits training data harder", font_size=23),
            Text("gamma small: smooth influence", font_size=23),
            Text("gamma large: local, detailed boundary", font_size=23),
        ).arrange(np.array([0, -1, 0]), aligned_edge=np.array([-1, 0, 0]), buff=0.22).shift((1.6, 0.65, 0))
        arrow = Arrow(start=(-0.6, -1.4, 0), end=(0.9, -1.4, 0), color=YELLOW)
        start = Text("2D nonlinear data", font_size=24).next_to(arrow, np.array([-1, 0, 0]))
        end = Text("SVM decision function", font_size=24).next_to(arrow, np.array([1, 0, 0]))
        final = Text("SVM finds a maximum-margin separator; kernels make nonlinear boundaries possible.", font_size=26, color=WHITE).to_edge(np.array([0, -1, 0]))

        self.play(Write(title))
        self.play(FadeIn(left), FadeIn(right))
        self.play(GrowArrow(arrow), FadeIn(start), FadeIn(end))
        self.play(FadeIn(final))
        self.wait(5)
        self.play(FadeOut(title), FadeOut(left), FadeOut(right), FadeOut(arrow), FadeOut(start), FadeOut(end), FadeOut(final))

    def _make_2d_axes(self):
        x_axis = Line((-3.2, 0, 0), (3.2, 0, 0), color=GREY_D)
        y_axis = Line((0, -2.2, 0), (0, 2.2, 0), color=GREY_D)
        x_label = Text("x1", font_size=20).next_to(x_axis, np.array([1, 0, 0]))
        y_label = Text("x2", font_size=20).next_to(y_axis, np.array([0, 1, 0]))
        return VGroup(x_axis, y_axis, x_label, y_label)

    def _make_ring_points(self):
        angles_inner = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        angles_outer = np.linspace(0, 2 * np.pi, 28, endpoint=False)
        inner = np.column_stack([0.55 * np.cos(angles_inner), 0.55 * np.sin(angles_inner)])
        outer = np.column_stack([1.55 * np.cos(angles_outer), 1.55 * np.sin(angles_outer)])
        points = np.vstack([inner, outer])
        labels = np.array([0] * len(inner) + [1] * len(outer))
        return points, labels

    def _dots_2d(self, points, labels):
        dots = VGroup()
        for point, label in zip(points, labels):
            color = BLUE if label == 0 else ORANGE
            dots.add(Dot(point=(point[0], point[1], 0), color=color, radius=0.065))
        return dots

    def _dots_3d(self, points, labels):
        dots = VGroup()
        for point, label in zip(points, labels):
            color = BLUE if label == 0 else ORANGE
            z = point[0] ** 2 + point[1] ** 2
            dots.add(Dot(point=(point[0], point[1], z), color=color, radius=0.055))
        return dots


class NonlinearDataProblem(Scene):
    """Short 2D-only scene for quick rendering."""

    def construct(self):
        title = Text("SVM: nonlinear data problem", font_size=38)
        formula = Text("w^T x + b = 0", font_size=38).next_to(title, np.array([0, -1, 0]))
        self.play(Write(title), Write(formula))
        self.wait(1)
