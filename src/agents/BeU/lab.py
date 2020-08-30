import grid2op

if __name__ == "__main__":
    env = grid2op.make("l2rpn_neurips_2020_track2_small")
    action_space = env.action_space
    print(action_space.size())
    try:
        do_nothing = action_space({})
        for i in range(200):
            # env.reset()
            done = True
            while not done:
                obs, reward, done, info = env.step(do_nothing)
                mix_name = env.name
                chronic_name = env.chronics_handler.get_name()
                print(
                    f"Episode [{i}] - Mix [{mix_name}] - Chronic [{chronic_name} - {obs.year}-{obs.month}-{obs.day} "
                    f"{obs.hour_of_day}:{obs.minute_of_hour}]")
    finally:
        env.close()
