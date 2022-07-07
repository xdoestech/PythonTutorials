#object with firstname, lastname, salary(pay), email
class Employee:
    #class variable
    num_of_emps = 0
    raise_amount = 1.04

    #initializer 
    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.pay = pay
        self.email = first + '.' + last + '@company.com'
        #class variable that increments every time
        print("added new employee {}".format(self.fullname()))
        Employee.num_of_emps += 1

    #returns the first and last name of employee as a string
    def fullname(self):
        return '{} {}'.format(self.first, self.last)
    #uses class variable to increase pay value
    def apply_raise(self):
        self.pay = int(self.pay * self.raise_amount)

#create TWO Employees
print('number of employees {}'.format(Employee.num_of_emps))
emp_1 = Employee('Xavier', 'Henschel', 1000000)
emp_2 = Employee('Test', 'User', 60000)
print('number of employees {}'.format(Employee.num_of_emps))

print(emp_1.pay)
Employee.apply_raise(emp_1)
print(emp_1.pay)
#print object fields as dictionary
print(emp_1.__dict__)

#change class
print("--change class raise amount--")
Employee.raise_amount = 1.05
print(emp_1.raise_amount)
print(emp_2.raise_amount)
#change instance
print('--change instance raise amount--')
emp_1.raise_amount = 1.69
print(emp_1.raise_amount)
print(emp_2.raise_amount)