#object with firstname, lastname, salary(pay), email
#how to use class and static methods 
    #Regular methods
        #pass self (instance) automatically
    #class methods
        #pass class automatically
    #static methods
        #do not pass anything automatically
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
#--------------REGULAR METHODS----------------------------
    #returns the first and last name of employee as a string
    def fullname(self):
        return '{} {}'.format(self.first, self.last)
    #uses class variable to increase pay value
    def apply_raise(self):
#--------------CLASS METHODS----------------------------
        self.pay = int(self.pay * self.raise_amount)
    #decorator recieve class instead of instance
    @classmethod
    def set_raise_amt(cls, amount):
        cls.raise_amount = amount
    #CLASS METHOD TO CREATE OBJECT
    @classmethod
    def from_string(cls, emp_str):
        first, last, pay = emp_str.split('-')
        return cls(first, last, pay)
#--------------STATIC METHODS----------------------------
    #returns False if saturday or sunday
    #does not access instance or class
    @staticmethod
    def is_weekday(day):
        if day.weekday() == 5 or  day.weekday() == 6:
            return False
        return True
#create TWO Employees
emp_1 = Employee('Xavier', 'Henschel', 1000000)
emp_2 = Employee('Test', 'User', 60000)

#change class
print("--change class raise amount (1.00)--")
Employee.raise_amount = 1.00
print(Employee.raise_amount)
print(emp_1.raise_amount)
print(emp_2.raise_amount)
#change instance
print('--change instance raise amount(1.69)--')
emp_1.raise_amount = 1.69
print(Employee.raise_amount)
print(emp_1.raise_amount)
print(emp_2.raise_amount)
print('----use class method to set raise amount(1.05)----')
Employee.set_raise_amt(1.05)
print(Employee.raise_amount)
print(emp_1.raise_amount)
print(emp_2.raise_amount)

#using class method to take in string 
emp_str_1 = 'John-Doe-70000'
emp_str_2 = 'Jane-Hoe-40000'
emp_str_3 = 'Steve-Smith-40000'
#create employee from string without class variable 
first, last, pay = emp_str_1.split('-')
new_emp_1 = Employee(first, last, pay)
#create employee from string with class variable
    #using CLASS VARIABLE to create object
    #alternative constructor
new_emp_2 = Employee.from_string(emp_str_3)
print(new_emp_1.email)
print(new_emp_2.email)
#test static method is_weekday
import datetime
my_date = datetime.date(2022, 6, 16)

print(Employee.is_weekday(my_date))