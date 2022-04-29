from .example import example


def capitalize_string(s):
    if not isinstance(s, str):
        raise TypeError('Please provide a string')
    return s.capitalize()


def test_capitalize_string():
    assert example.capitalize_string('test') == 'Test'