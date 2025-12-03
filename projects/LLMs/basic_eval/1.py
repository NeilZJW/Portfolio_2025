# print("Hello World!")
#
# print("Neil says: 'Hazel is a fool!'")
# print('Neil says: "Hazel is a fool!"')
# print("Neil says: \"Hazel is a fool!\"")
#
# print("""
#
# 1232
#     12312
#     1551515
#     1111114
#                     2424
# """)

# a = input("Please enter your password: ")
# print("Your password is: ", a)

# a = 1
# b = 1.0
# c = "1"
#
# print("a的数据类型是： ", type(a))
# print("b的数据类型是： ", type(b))
# print("c的数据类型是： ", type(c))
#
# a = float(1)
# b = str(1.0)
# c = int("1")
#
# print("a的数据类型是： ", type(a))
# print("b的数据类型是： ", type(b))
# print("c的数据类型是： ", type(c))


# This is a single comment
"""
This is also a comment
but multiple lines
"""


# 需要用户输入密码，并且判断是否正确，并打印相关信息
real_pw = 123456
input_pw = int(input("Please enter your password: "))
if real_pw == input_pw:
    print("Welcome!")
else:
    print("Invalid password!")



