

def dealDefault(default):
    default = default.replace("mstype", "mindspore")
    i = default.strip()
    if len(i) == 0:
        i = ""
    else:
        i = i.replace("\"", "")
        i = i.replace("\'", "")
        i = i.replace("‘", "")
        i = i.replace("’", "")
        i = i.replace("“", "")
        i = i.replace("”", "")
    return i
