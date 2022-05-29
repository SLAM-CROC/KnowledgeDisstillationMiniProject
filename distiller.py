import tensorflow as tf


class Distiller(tf.keras.Model):
    # The constructor of distiller
    def __init__(self, student,  teacher0, teacher1, teacher2, optimizer,
                 student_loss_fn, distillation_loss_fn, acc_metrics, alpha, temperature):
        super(Distiller, self).__init__()
        self.teacher0 = teacher0
        self.teacher1 = teacher1
        self.teacher2 = teacher2
        self.student = student
        self.optimizer = optimizer
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.acc_metrics = acc_metrics
        self.alpha = alpha
        self.temperature = temperature

    # Customized train loop for KD
    def train_student(self, num_epochs, dataset_domain0, dataset_domain1, dataset_domain2):
        for epoch in range(num_epochs):
            print(f"\nStart of Training Epoch {epoch}")

            # KD in domain 0
            for batch_idx, (x_batch, y_batch) in enumerate(dataset_domain0):
                # Forward pass of teacher
                teacher_predictions = self.teacher0(x_batch, training=False)

                with tf.GradientTape() as tape:
                    # Forward pass of student
                    student_predictions = self.student(x_batch, training=True)
                    student_predictions_2 = self.student(x_batch, training=False)

                    # Compute losses
                    student_loss = self.student_loss_fn(y_batch, student_predictions)
                    distillation_loss = self.distillation_loss_fn(
                        tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                        tf.nn.softmax(student_predictions_2 / self.temperature, axis=1),
                    )
                    loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

                # Compute gradients
                trainable_vars = self.student.trainable_variables
                gradients = tape.gradient(loss, trainable_vars)

                # Update weights
                self.optimizer.apply_gradients(zip(gradients, trainable_vars))

                # Update the metrics
                self.acc_metrics.update_state(y_batch, student_predictions)

            # KD in domain 1
            for batch_idx, (x_batch, y_batch) in enumerate(dataset_domain1):
                teacher_predictions = self.teacher1(x_batch, training=False)

                with tf.GradientTape() as tape:
                    # Forward pass of student
                    student_predictions = self.student(x_batch, training=True)
                    student_predictions_2 = self.student(x_batch, training=False)

                    # Compute losses
                    student_loss = self.student_loss_fn(y_batch, student_predictions)
                    distillation_loss = self.distillation_loss_fn(
                        tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                        tf.nn.softmax(student_predictions_2 / self.temperature, axis=1),
                    )
                    loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

                # Compute gradients
                trainable_vars = self.student.trainable_variables
                gradients = tape.gradient(loss, trainable_vars)

                self.optimizer.apply_gradients(zip(gradients, trainable_vars))

                self.acc_metrics.update_state(y_batch, student_predictions)

            # KD in domain 2
            for batch_idx, (x_batch, y_batch) in enumerate(dataset_domain2):
                teacher_predictions = self.teacher2(x_batch, training=False)

                with tf.GradientTape() as tape:
                    # Forward pass of student
                    student_predictions = self.student(x_batch, training=True)
                    student_predictions_2 = self.student(x_batch, training=False)

                    # Compute losses
                    student_loss = self.student_loss_fn(y_batch, student_predictions)
                    distillation_loss = self.distillation_loss_fn(
                        tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                        tf.nn.softmax(student_predictions_2 / self.temperature, axis=1),
                    )
                    loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

                # Compute gradients
                trainable_vars = self.student.trainable_variables
                gradients = tape.gradient(loss, trainable_vars)

                self.optimizer.apply_gradients(zip(gradients, trainable_vars))

                self.acc_metrics.update_state(y_batch, student_predictions)

            # Print training accuracy at each end of epoch
            train_acc = self.acc_metrics.result()
            print(f"Accuracy over epoch {train_acc}")
            self.acc_metrics.reset_states()

    # Customized test loop for KD
    def test_student(self, dataset):
        for batch_idx, (x_batch, y_batch) in enumerate(dataset):
            y_pred = self.student(x_batch, training=True)
            self.acc_metrics.update_state(y_batch, y_pred)

        train_acc = self.acc_metrics.result()
        print(f"Accuracy over Test Set: {train_acc}")
        self.acc_metrics.reset_states()
        return train_acc
