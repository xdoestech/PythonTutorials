#object with firstname, lastname, salary(pay), email
from multiprocessing import managers


class Employee:
    #class variable
    num_of_emps = 0
    raise_amount = 1.04

    #initializer 
    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.pay = pay
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

#-----------------SUBCLASS--------------------  
    def fullname(self):
        return '{} {}'.format(self.first, self.last)
    #uses class variable to increase pay value
    def apply_raise(self):
        self.pay = int(self.pay * self.raise_amount)

#-----------------SUBCLASS--------------------  
#inheretes all attributes and methods from Employee Class
    #give init method to add more info
class Developer(Employee):
    raise_amount = 1.10
    #take in all Employee fields add programming language
    def __init__(self, first, last, pay, prog_lang):
        super().__init__(first, last, pay)
        self.prog_lang = prog_lang
#subclass taking in a list
    def __init__(self, first, last, pay, prog_lang):
        super().__init__(first, last, pay)
        self.prog_lang = prog_lang
#subclass taking in a list
class Manager(Employee):
    def __init__(self, first, last, pay, employees=None):
        super().__init__(first, last, pay)
        if employees is None:
            self.employees = []
    def __init__(self, first, last, pay, employees=None):
        super().__init__(first, last, pay)
        if employees is None:
            self.employees = []
        else:
            self.employees = employees

    def add_emp(self, emp):
        if emp not in self.employees:
            self.employees.append(emp)
    
    def remove_emp(self, emp):
        if emp not in self.employees:
            self.employees.remove(emp)

    def print_emps(self):
        for emp in self.employees:
            print('-->', emp.fullname())
    
dev_1 = Developer('Xavier', 'Henschel', 50000, 'python')
dev_2 = Employee('Test', 'Employee', 20000)

mgr_1 = Manager('Sue', 'Smith', 90000, [dev_1])
print(mgr_1.email)
mgr_1.print_emps()

print(dev_1.email)
print(dev_1.prog_lang)

#----------HELPFUL FUNCTIONS -------------
    #-----HELP CLASS---------
        #prints info about class
    #print(help(Developer))
    #------isinstance-------
        #prints true if object is an instance of an object type
        #includes superclass/inheretance 
        #(manager isintance of manager and employee)
    #print(isinstance(mgr_1, Manager))
    #------issubclass------
        #prints true if object is subclass of another object
        #print(issubclass(Manager, Developer))











