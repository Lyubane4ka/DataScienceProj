def print_students_info(students):
    """
     Выводит информацию о студентах в формате "Имя: [имя], Возраст: [возраст], Курс: [курс]"
    :param students: (list) Список словарей с информацией о студентах
    :return: students info
    """
    if students is not None:
        for student in students:
            name = student.get('имя', 'Не указано')
            age = student.get('возраст', 'Не указан')
            course = student.get('курс', 'Не указан')
            print(f"Имя: {name}, Возраст: {age}, Курс: {course}")
        else:
            print("Данные отсутствуют")
