ACTION_INDICES = [0, 1, 2, 3]

ACTION_NAMES = ['ArrowUp', 'ArrowLeft', 'ArrowRight', 'n']

ACTION_VALUES = [
    [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False),
     ('KeyEvent', 'n', False)],
    [('KeyEvent', 'ArrowUp', False), ('KeyEvent', 'ArrowLeft', True), ('KeyEvent', 'ArrowRight', False),
     ('KeyEvent', 'n', False)],
    [('KeyEvent', 'ArrowUp', False), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', True),
     ('KeyEvent', 'n', False)],
    [('KeyEvent', 'ArrowUp', False), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False),
     ('KeyEvent', 'n', True)]
]
