class Student:
    def __init__(self, averageMark, firstName, lastName, group):
        self.averageMark = averageMark
        self.firstName = firstName
        self.lastName = lastName
        self.group = group
    def getScholarship1(self):
        if self.averageMark == 5:
            return 5000
        else:
            return 0
    def display(self):
        print(f"****************\n Имя фамилия: {self.firstName} {self.lastName}\n Группа: {self.group}\n Средний балл: {self.averageMark}\n Класс: Студент\n Стипендия: {self.getScholarship1}\n****************")

class Aspirant(Student):
    def getScholarship2(self):
        if self.averageMark == 5:
            return 6000
        else:
            return 0

    def display(self):
        print(f"****************\n Имя фамилия: {self.firstName} {self.lastName}\n Группа {self.group}\n Средний балл: {self.averageMark}\n Класс: Аспирант\n Стипендия: {self.getScholarship2}\n****************")

firstName = input()
lastName = input()
group = input()
averageMark = float(input())

student = Student(averageMark, firstName, lastName, group)
aspirant = Aspirant(averageMark, firstName, lastName, group)
getScholarship1 = student.getScholarship1()
getScholarship2 = aspirant.getScholarship2()

student.display()
aspirant.display()
