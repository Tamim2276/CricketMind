from src.utils import progress


def test_the_bar_fills_up():
    assert progress.dots(0, 10).count(chr(0x25CF)) == 0
    assert progress.dots(5, 10).count(chr(0x25CF)) == progress.WIDTH // 2
    assert progress.dots(10, 10).count(chr(0x25CF)) == progress.WIDTH


def test_the_bar_is_always_the_same_width():
    for done in range(0, 11):
        glyphs = (progress.dots(done, 10).count(chr(0x25CF))
                  + progress.dots(done, 10).count(chr(0xB7)))
        assert glyphs == progress.WIDTH


def test_a_terminal_line_rewrites_itself():
    out = progress.line(3, 10, "tail", tty=True)
    assert out.startswith(chr(13)) and "\n" not in out


def test_a_log_line_has_no_escape_codes():
    out = progress.line(3, 10, "tail", tty=False)
    assert "\033" not in out and chr(13) not in out
    assert "3/10" in out and "tail" in out


def test_zero_total_does_not_divide_by_zero():
    assert progress.line(0, 0, tty=True)
    assert progress.dots(0, 0).count(chr(0x25CF)) == progress.WIDTH


def test_overshooting_does_not_overflow_the_bar():
    assert progress.dots(20, 10).count(chr(0x25CF)) == progress.WIDTH
