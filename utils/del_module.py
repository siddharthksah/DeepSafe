import sys

def delete_module(modname, paranoid=None):
    from sys import modules
    
    try:
        thismod = modules[modname]
    except KeyError:
        raise ValueError(f"Module {modname} not found")

    these_symbols = dir(thismod)
    
    if paranoid:
        if not isinstance(paranoid, (list, tuple)):
            raise ValueError('Paranoid must be a finite list or tuple')
        these_symbols = paranoid[:]
    
    del modules[modname]
    
    for mod in modules.values():
        try:
            delattr(mod, modname)
        except AttributeError:
            pass
        
        if paranoid:
            for symbol in these_symbols:
                if symbol.startswith('__'):  # ignore special symbols
                    continue
                try:
                    delattr(mod, symbol)
                except AttributeError:
                    pass
