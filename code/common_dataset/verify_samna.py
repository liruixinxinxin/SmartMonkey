import samna



d = samna.device.get_unopened_devices()
dk = samna.device.open_device(d[0])
input_node = samna.graph.source_to(dk.get_model_sink_node())
buf = samna.graph.sink_from(dk.get_model_source_node())


input_node.write([samna.xyloImu.event.WriteRegisterValue(0, 10)])