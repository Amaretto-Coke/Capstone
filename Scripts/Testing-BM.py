def get_pkmn_pallet(hues=None, types=None):
    pkmn_colors = {
        'Bug':      {'light': 'C6D16E', 'regular': 'A8B820', 'dark': '6D7815'},
        'Dark':     {'light': 'A29288', 'regular': '705848', 'dark': '49392F'},
        'Dragon':   {'light': 'A27DFA', 'regular': '7038F8', 'dark': '4924A1'},
        'Electric': {'light': 'FAE078', 'regular': 'F8D030', 'dark': 'A1871F'},
        'Fairy':    {'light': 'F4BDC9', 'regular': 'EE99AC', 'dark': '9B6470'},
        'Fighting': {'light': 'D67873', 'regular': 'C03028', 'dark': '7D1F1A'},
        'Fire':     {'light': 'F5AC78', 'regular': 'F08030', 'dark': '9C531F'},
        'Flying':   {'light': 'C6B7F5', 'regular': 'A890F0', 'dark': '6D5E9C'},
        'Ghost':    {'light': 'A292BC', 'regular': '705898', 'dark': '493963'},
        'Grass':    {'light': 'A7DB8D', 'regular': '78C850', 'dark': '4E8234'},
        'Ground':   {'light': 'EBD69D', 'regular': 'E0C068', 'dark': '927D44'},
        'Ice':      {'light': 'BCE6E6', 'regular': '98D8D8', 'dark': '638D8D'},
        'Normal':   {'light': 'C6C6A7', 'regular': 'A8A878', 'dark': '6D6D4E'},
        'Poison':   {'light': 'C183C1', 'regular': 'A040A0', 'dark': '682A68'},
        'Psychic':  {'light': 'FA92B2', 'regular': 'F85888', 'dark': 'A13959'},
        'Rock':     {'light': 'D1C17D', 'regular': 'B8A038', 'dark': '786824'},
        'Steel':    {'light': 'D1D1E0', 'regular': 'B8B8D0', 'dark': '787887'},
        'Water':    {'light': '9DB7F5', 'regular': '6890F0', 'dark': '445E9C'},
        '???':      {'light': '9DC1B7', 'regular': '68A090', 'dark': '44685E'}
    }

    if hues is not None and types is not None:
        # If the func call specifies both the type(s) and the hues
        result = pkmn_colors
        pop_list = []
        for key in result.keys():
            if key not in types:
                pop_list.append(key)
        for key in pop_list:
            result.pop(key)
        for type_hue in result:
            x = result[type_hue]
            pop_list = []
            for y in x:
                if y not in hues:
                    pop_list.append(y)
            for y in pop_list:
                x.pop(y)
            result.pop(type_hue)
            result[type_hue] = x
        print(result)
    elif hues is not None:
        # If the func call specifies the hues but not the type
        pass
    elif types is not None:
        pass
    else:
        pass

get_pkmn_pallet(hues=['light'], types=['Water', 'Dark'])
