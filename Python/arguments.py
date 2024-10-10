# Positional Arguments

def introduce(name, age):
    print(f"My name is {name} and I am {age} years old.")

introduce("Alice", 30)  # Positional arguments

#default arguments

def introduce(name, age=25):  # 'age' has a default value of 25
    print(f"My name is {name} and I am {age} years old.")

introduce("Bob")          # Uses the default value for 'age'
introduce("Charlie", 40)

#keword arguments
def introduce(name, age):
    
    print(f"My name is {name} and I am {age} years old.")  # Keyword arguments
introduce(name="Alice", age=30) 





