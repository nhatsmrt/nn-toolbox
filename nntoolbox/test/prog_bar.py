def progress_bar_test():
    from fastprogress import master_bar, progress_bar
    from fastprogress.fastprogress import NBMasterBar
    from time import sleep, time
    print("Starting progress bar test")

    mb = master_bar(range(500))
    print(isinstance(mb, NBMasterBar))
    mb.on_iter_begin()
    pb = progress_bar(range(100), parent=mb, auto_update=False)
    mb.update(0)
    iter_cnt = 0

    for e in range(500):
        for _ in range(100):
            sleep(0.1)
            iter = iter_cnt % len(range(100))
            pb.update(iter)
            iter_cnt += 1
        pb = progress_bar(range(100), parent=mb, auto_update=False)
        #   mb.write([format_time(time() - start)], table=True)
        mb.update(e + 1)
    mb.on_iter_end()