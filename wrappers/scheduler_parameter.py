




##### Scheduler parameters example :
def regular_scheduler(temperature, epoch, cste = 0.999):
    if temperature < 1e-2 :
        return temperature
    return temperature * cste

