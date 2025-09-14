def print_students_info(students):
    """
     Выводит информацию о студентах в формате "Имя: [имя], Возраст: [возраст], Курс: [курс]"
    :param students: (list) Список словарей с информацией о студентах
    :return: students info
    """
    if students is not None:  # O(1)
        for student in students:  # O(n)
            name = student.get('имя', 'Не указано')  # O(1) - доступ по ключу в словаре
            age = student.get('возраст', 'Не указан')  # O(1)
            course = student.get('курс', 'Не указан')  # O(1)
            print(f"Имя: {name}, Возраст: {age}, Курс: {course}")  # O(1)-вывод 1 строки
        else:
            print("Данные отсутствуют")  # в случае выполнения O(1)

# общая временная сложность алгоритма O(n)
