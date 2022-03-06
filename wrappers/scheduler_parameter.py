




##### Scheduler parameters example :
def regular_scheduler(temperature, epoch, cste = 0.99999):
    if temperature < 1e-5 :
        return temperature
    return temperature * cste

