

if __name__ == "__main__":

    import numpy as np
    from model_control.model_controller.retrain_clf.controller import ModelController
    n_sample = 30
    input_size = 51200

    train_x = np.random.random((n_sample, input_size))
    train_y = np.random.choice(2, n_sample)

    valid_x = np.random.random((n_sample, input_size))
    valid_y = np.random.choice(2, n_sample)

    model_controller = ModelController(train_x, train_y, valid_x, valid_y)
    model_controller.read_config("config.json")
    model_controller.build()
    # model_controller.load_weight(
    #     "test_model_state_dict.pth")
    model_controller.compile(model_name="change_model_name")
    model_controller.train(5)
    y_pred = model_controller.predict(train_x, batch_size=5)
    model_controller.save()
    score = model_controller.evaluate(train_y, y_pred)

    print(model_controller.model_name)
    print(model_controller.model)
    print(model_controller.train_loader)
    print(model_controller.valid_loader)
    print(score)
