def test_model_output_shape():
    """Test if model outputs correct shape"""
    model = MultiSourceUNet()
    x = torch.randn(1, 8, 256, 256)
    out = model(x)
    assert out.shape == (1, 1, 256, 256)

def test_model_training():
    """Test if model trains properly"""
    model = MultiSourceUNet()
    criterion = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters())
    # Add training test 