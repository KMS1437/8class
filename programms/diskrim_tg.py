import math
import telebot

TOKEN = input("Введите токен от бота, который вы получите в боте BotFather: ")
bot = telebot.TeleBot(TOKEN)


def solve_quadratic(a, b, c):
    result = []

    if a == 0 and b == 0 and c == 0:
        result.append("Программа завершена.")
        return "\n".join(result)

    result.append(f"Рассматриваем уравнение: {a}x^2 + {b}x + {c} = 0")

    if b == 0 and c == 0:
        result.append("Уравнение неполное (b = 0 и c = 0).")
        result.append("Оно имеет единственный корень: x = 0.")
        result.append("Ответ: 0")
        return "\n".join(result)

    if b == 0:
        result.append("Уравнение неполное (b = 0).")
        result.append(f"{a}x^2 + {c} = 0")
        if -c / a < 0:
            result.append("Подкоренное выражение отрицательно, корней нет.")
            result.append("Ответ: Корней нет")
        else:
            x1 = math.sqrt(-c / a)
            x2 = -math.sqrt(-c / a)
            result.append(f"Корни: x1 = {x1}, x2 = {x2}")
            result.append(f"Ответ: {x1}; {x2}")
        return "\n".join(result)

    if c == 0:
        result.append("Уравнение неполное (c = 0).")
        result.append(f"{a}x^2 + {b}x = 0")
        x1 = 0
        x2 = -b / a
        result.append(f"Корни: x1 = {x1}, x2 = {x2}")
        result.append(f"Ответ: {x1}; {x2}")
        return "\n".join(result)

    if a + b + c == 0:
        result.append(f"a + b + c = 0 ({a} + {b} + {c}), корни: x1 = 1, x2 = {c / a}")
        result.append(f"Ответ: 1; {c / a}")
        return "\n".join(result)

    if a - b + c == 0:
        result.append(f"a - b + c = 0 ({a} - {b} + {c}), корни: x1 = -1, x2 = {-c / a}")
        result.append(f"Ответ: -1; {-c / a}")
        return "\n".join(result)

    D = b ** 2 - 4 * a * c
    result.append(f"Дискриминант: D = {b}^2 - 4 * {a} * {c} = {D}")

    if D > 0:
        x1 = (-b + math.sqrt(D)) / (2 * a)
        x2 = (-b - math.sqrt(D)) / (2 * a)
        result.append("Дискриминант положительный, два корня.")
        result.append(f"x1 = {x1}, x2 = {x2}")
        result.append(f"Ответ: {x1}; {x2}")
    elif D == 0:
        x1 = -b / (2 * a)
        result.append("Дискриминант равен нулю, один корень.")
        result.append(f"x1 = {x1}")
        result.append(f"Ответ: {x1}")
    else:
        result.append("Дискриминант отрицательный, корней нет.")
        result.append("Ответ: Корней нет")

    return "\n".join(result)


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Отправь коэффициенты a, b и c через пробел, и я решу квадратное уравнение.")


@bot.message_handler(func=lambda message: True)
def handle_message(message):
    try:
        a, b, c = map(int, message.text.split())
        response = solve_quadratic(a, b, c)
        bot.reply_to(message, response)
    except ValueError:
        bot.reply_to(message, "Пожалуйста, введите три целых числа через пробел.")


print("Бот запущен...")
bot.polling()
