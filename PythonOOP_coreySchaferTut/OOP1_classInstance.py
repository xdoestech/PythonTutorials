#attributes/Methods (data/functions associated with class)

class EmployeeP:
    pass #skips class

#class instances
emp_1 = EmployeeP()
emp_2 = EmployeeP()

print(emp_1)
print(emp_2)

emp_1.first = 'Xavier'
emp_1.last = 'Henschel'
emp_1.email = 'xavierh21@vt.edu'
emp_1.pay = 1000000

emp_2.first = 'Test'
emp_2.last = 'User'
emp_2.email = 'Test.User@vt.edu'
emp_2.pay = 20000

print(emp_2.email)
print(emp_1.email)

#initialize class COMMENT DESCRIPTION
    #creates employee 
class Employee:
    #init method 
    def __init__(self, first, last, pay) -> None: #specifies return type of none
        self.first = first
        self.last = last 
        self.pay = pay 
        self.email = first + '.' + last + '@company.com'
    #returns the full name of the employee
    def fullname(self):
        return '{} {}'.format(self.first, self.last)

emp_a = Employee('Jimmy', 'Buffet', 10000)
emp_b = Employee('Killa', 'Kimbo', 6942069)

print(emp_a.fullname())
print(Employee.fullname(emp_a)) #preffered way to do it for readability

