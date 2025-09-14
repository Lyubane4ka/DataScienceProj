import unittest

from Task19 import print_students_info


class TestStudentInfo(unittest.TestCase):

    def test_task19(self):
        students_list = [
            {'имя': 'Иван', 'возраст': 20, 'курс': 2},
            {'имя': 'Мария', 'возраст': 19, 'курс': 1},
            {'имя': 'Петр', 'возраст': 21, 'курс': 3},
            {'имя': 'Анна', 'возраст': 22}  # Отсутствует курс
        ]
        print("Информация о студентах:")
        print_students_info(students_list)

    def test_task19_emptyList(self):
        print("Информация о студентах:")
        print_students_info(None)
