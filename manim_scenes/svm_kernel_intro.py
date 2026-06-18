"""Manim phase-1 concept animation for the SVM kernel trick.

The scene follows the phase-1 frame in ``SVM_3D教學設計概念圖表.png``:
2D nonlinear data -> feature mapping -> 3D lift -> separating plane ->
projected nonlinear boundary -> teaching summary.
"""

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
    Circle,
    Dot,
    FadeIn,
    FadeOut,
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
    """A compact under-30-second teaching animation for the kernel trick."""

    def construct(self):
        self.camera.background_color = "#0b1220"
        points, labels = self._make_ring_points()

        title = Text("SVM Kernel Trick: 2D to 3D", font_size=40, color=BLUE).to_edge(np.array([0, 1, 0]))
        subtitle = Text("Goal: make nonlinear data separable", font_size=25, color=WHITE).next_to(title, np.array([0, -1, 0]), buff=0.25)
        self.add_fixed_in_frame_mobjects(title, subtitle)
        self.play(Write(title), FadeIn(subtitle), run_time=1.5)
        self.wait(0.5)
        self.play(FadeOut(subtitle), run_time=0.4)

        axes2 = self._make_2d_axes()
        dots2 = self._dots_2d(points, labels)
        note = Text("1. In 2D, one straight line cannot separate the rings.", font_size=24, color=WHITE).to_edge(np.array([0, -1, 0]))
        bad_line = Line((-2.2, -1.6, 0), (2.2, 1.6, 0), color=GREY_B)
        self.play(Create(axes2), FadeIn(dots2), FadeIn(note), run_time=1.6)
        self.play(Create(bad_line), run_time=0.8)
        self.wait(0.8)

        formula = Text("2. Feature map:  phi(x, y) = (x, y, x^2 + y^2)", font_size=25, color=YELLOW).to_edge(np.array([0, -1, 0]))
        self.play(FadeOut(bad_line), FadeOut(note), FadeIn(formula), run_time=1.0)
        self.wait(0.8)
        self.play(FadeOut(axes2), FadeOut(dots2), FadeOut(formula), FadeOut(title), run_time=0.8)

        axes3 = ThreeDAxes(
            x_range=(-2.2, 2.2),
            y_range=(-2.2, 2.2),
            z_range=(0, 3.0),
            x_length=5.3,
            y_length=5.3,
            z_length=3.0,
        )
        mapped = self._dots_3d(points, labels)
        surface = Surface(
            lambda u, v: np.array([u, v, u * u + v * v]),
            u_range=(-1.7, 1.7),
            v_range=(-1.7, 1.7),
            resolution=(18, 18),
            fill_opacity=0.22,
            checkerboard_colors=[BLUE, PURPLE],
        )
        lift_note = Text("3. Lift points upward by radius: z = x^2 + y^2", font_size=24, color=WHITE).to_edge(np.array([0, -1, 0]))
        self.add_fixed_in_frame_mobjects(lift_note)

        self.set_camera_orientation(phi=62 * DEGREES, theta=-42 * DEGREES)
        self.play(Create(axes3), FadeIn(mapped), FadeIn(lift_note), run_time=1.8)
        self.play(FadeIn(surface), run_time=1.2)
        self.begin_ambient_camera_rotation(rate=0.12)
        self.wait(2.0)

        plane = Rectangle(width=4.4, height=4.4, color=GREEN, fill_color=GREEN, fill_opacity=0.24)
        plane.rotate(90 * DEGREES, axis=np.array([1, 0, 0])).shift((0, 0, 1.15))
        plane_note = Text("4. In feature space, a plane can separate the classes.", font_size=24, color=GREEN).to_edge(np.array([0, -1, 0]))
        self.add_fixed_in_frame_mobjects(plane_note)
        self.play(Create(plane), FadeOut(lift_note), FadeIn(plane_note), run_time=1.5)
        self.wait(2.0)
        self.stop_ambient_camera_rotation()
        self.play(FadeOut(axes3), FadeOut(mapped), FadeOut(surface), FadeOut(plane), FadeOut(plane_note), run_time=0.8)

        self.move_camera(phi=0 * DEGREES, theta=-90 * DEGREES, run_time=0.4)
        axes_back = self._make_2d_axes()
        dots_back = self._dots_2d(points, labels)
        boundary = Circle(radius=1.05, color=YELLOW).set_stroke(width=5)
        margin_inner = Circle(radius=0.88, color=WHITE).set_stroke(width=2, opacity=0.7)
        margin_outer = Circle(radius=1.22, color=WHITE).set_stroke(width=2, opacity=0.7)
        support = self._support_marks()
        back_note = Text("5. Project back: the plane becomes a nonlinear boundary.", font_size=24, color=YELLOW).to_edge(np.array([0, -1, 0]))
        self.add_fixed_in_frame_mobjects(back_note)
        self.play(Create(axes_back), FadeIn(dots_back), Create(boundary), FadeIn(back_note), run_time=1.7)
        self.play(Create(margin_inner), Create(margin_outer), FadeIn(support), run_time=1.2)
        self.wait(1.8)

        self.play(FadeOut(axes_back), FadeOut(dots_back), FadeOut(boundary), FadeOut(margin_inner), FadeOut(margin_outer), FadeOut(support), FadeOut(back_note), run_time=0.7)

        summary = VGroup(
            Text("Key idea", font_size=34, color=BLUE),
            Text("Kernel trick does not draw a magic curve first.", font_size=24, color=WHITE),
            Text("It compares points in a richer feature space.", font_size=24, color=WHITE),
            Text("SVM then finds the maximum-margin separator.", font_size=24, color=YELLOW),
        ).arrange(np.array([0, -1, 0]), buff=0.25).move_to((0, -0.15, 0))
        warning = Text("Teaching map: z = x^2 + y^2. Real RBF is implicit and higher-dimensional.", font_size=20, color=RED).to_edge(np.array([0, -1, 0]))
        self.add_fixed_in_frame_mobjects(summary, warning)
        self.play(FadeIn(summary), FadeIn(warning), run_time=1.4)
        self.wait(3.0)
        self.play(FadeOut(summary), FadeOut(warning), run_time=0.8)

    def _make_2d_axes(self):
        x_axis = Line((-3.1, 0, 0), (3.1, 0, 0), color=GREY_D)
        y_axis = Line((0, -2.15, 0), (0, 2.15, 0), color=GREY_D)
        x_label = Text("x", font_size=18).next_to(x_axis, np.array([1, 0, 0]))
        y_label = Text("y", font_size=18).next_to(y_axis, np.array([0, 1, 0]))
        return VGroup(x_axis, y_axis, x_label, y_label)

    def _make_ring_points(self):
        inner_angles = np.linspace(0, 2 * np.pi, 14, endpoint=False)
        outer_angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        inner = np.column_stack([0.55 * np.cos(inner_angles), 0.55 * np.sin(inner_angles)])
        outer = np.column_stack([1.55 * np.cos(outer_angles), 1.55 * np.sin(outer_angles)])
        points = np.vstack([inner, outer])
        labels = np.array([0] * len(inner) + [1] * len(outer))
        return points, labels

    def _dots_2d(self, points, labels):
        dots = VGroup()
        for point, label in zip(points, labels):
            dots.add(Dot(point=(point[0], point[1], 0), color=BLUE if label == 0 else ORANGE, radius=0.065))
        return dots

    def _dots_3d(self, points, labels):
        dots = VGroup()
        for point, label in zip(points, labels):
            z = point[0] ** 2 + point[1] ** 2
            dots.add(Dot(point=(point[0], point[1], z), color=BLUE if label == 0 else ORANGE, radius=0.055))
        return dots

    def _support_marks(self):
        marks = VGroup()
        for radius, count in [(0.55, 4), (1.55, 6)]:
            for angle in np.linspace(0, 2 * np.pi, count, endpoint=False):
                point = (radius * np.cos(angle), radius * np.sin(angle), 0)
                marks.add(Circle(radius=0.13, color=YELLOW).move_to(point))
        return marks


class NonlinearDataProblem(Scene):
    """Short 2D-only scene for quick rendering checks."""

    def construct(self):
        title = Text("SVM: nonlinear data problem", font_size=38)
        formula = Text("w^T x + b = 0", font_size=38).next_to(title, np.array([0, -1, 0]))
        self.play(Write(title), Write(formula))
        self.wait(1)
