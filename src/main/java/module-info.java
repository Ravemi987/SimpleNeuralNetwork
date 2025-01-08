module fr.simpleneuralnetwork {
    requires javafx.controls;
    requires javafx.fxml;
    requires ejml.simple;
    requires ejml.core;

    exports fr.simpleneuralnetwork.main;
    opens fr.simpleneuralnetwork.main to javafx.fxml;
    exports fr.simpleneuralnetwork.view;
    opens fr.simpleneuralnetwork.view to javafx.fxml;
}