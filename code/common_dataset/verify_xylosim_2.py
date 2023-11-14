import samna  


def initialize_board():
    dk = samna.device.open_device("XyloImuTestBoard:0")  
    buf = samna.graph.sink_from(dk.get_model_source_node())  
    source = samna.graph.source_to(dk.get_model_sink_node())  
    return dk, buf, source  

dk, buf, source = initialize_board()  


def build_event_type_filters(dk, graph):
    _, etf0, register_value_buf = graph.sequential([dk.get_model_source_node(), "XyloImuOutputEventTypeFilter", samna.graph.JitSink()])
    etf0.set_desired_type('xyloImu::event::RegisterValue')
    _, etf1, readout_buf = graph.sequential([dk.get_model_source_node(), "XyloImuOutputEventTypeFilter", samna.graph.JitSink()])
    etf1.set_desired_type('xyloImu::event::Readout')
    _, etf2, interrupt_buf = graph.sequential([dk.get_model_source_node(), "XyloImuOutputEventTypeFilter", samna.graph.JitSink()])
    etf2.set_desired_type('xyloImu::event::Interrupt')
    _, etf3, membrane_potential_buf = graph.sequential([dk.get_model_source_node(), "XyloImuOutputEventTypeFilter", samna.graph.JitSink()])
    etf3.set_desired_type('xyloImu::event::MembranePotential')
    _, etf4, synaptic_current_buf = graph.sequential([dk.get_model_source_node(), "XyloImuOutputEventTypeFilter", samna.graph.JitSink()])
    etf4.set_desired_type('xyloImu::event::SynapticCurrent')
    _, etf5, hidden_spike_buf = graph.sequential([dk.get_model_source_node(), "XyloImuOutputEventTypeFilter", samna.graph.JitSink()])
    etf5.set_desired_type('xyloImu::event::HiddenSpikeCount')

    return register_value_buf, readout_buf, interrupt_buf, membrane_potential_buf, synaptic_current_buf, hidden_spike_buf  # 返回各种事件类型的缓冲区

graph = samna.graph.EventFilterGraph()  
register_value_buf, readout_buf, interrupt_buf, membrane_potential_buf, synaptic_current_buf, hidden_spike_buf = build_event_type_filters(dk, graph)  # 调用函数创建事件过滤器

graph.start()  


def read_register(address):
    buf.get_events()
    source.write([samna.xyloImu.event.ReadRegisterValue(address=address)])
    events = register_value_buf.get_n_events(1, 2000)  
    assert (len(events) == 1)
    return events[0].data


def trigger_processing():
    buf.get_events()
    source.write([samna.xyloImu.event.TriggerProcessing()])
    interrupt_events = interrupt_buf.get_n_events(1, 2000)  
    if not interrupt_events:
        raise Exception("No interrupt occurs after processing done!")

def request_readout(hidden_count, output_count):
    buf.get_events()
    source.write([samna.xyloImu.event.TriggerReadout()])
    readouts = readout_buf.get_n_events(1, 2000)

    # Only two attributes of `Readout` event is available in manual mode: `timestep`, `output_v_mems`.
    # We have to read all other things manually in manual mode.
    assert(len(readouts) == 1)
    readout = readouts[0]

    # Read all membrane potentials
    for _ in range(2):      # Due to a bug on chip, you have to read memory twice to ensure it's correct.
        source.write([samna.xyloImu.event.ReadMembranePotential(neuron_id = i) for i in range(hidden_count + output_count)])
        membrane_potentials = membrane_potential_buf.get_n_events(hidden_count + output_count, 5000)
        assert(len(membrane_potentials) == hidden_count + output_count)
        readout.neuron_v_mems = [e.value for e in membrane_potentials]

    # Read all synaptic current
    for _ in range(2):      # Due to a bug on chip, you have to read memory twice to ensure it's correct.
        source.write([samna.xyloImu.event.ReadSynapticCurrent(neuron_id = i) for i in range(hidden_count + output_count)])
        synaptic_currents = synaptic_current_buf.get_n_events(hidden_count + output_count, 5000)
        assert(len(synaptic_currents) == hidden_count + output_count)
        readout.neuron_i_syns = [e.value for e in synaptic_currents]

    # Read all hidden spike count
    source.write([samna.xyloImu.event.ReadHiddenSpikeCount(neuron_id = i) for i in range(hidden_count)])
    hidden_spikes = hidden_spike_buf.get_n_events(hidden_count, 5000)
    assert(len(hidden_spikes) == hidden_count)
    readout.hidden_spikes = [e.count for e in hidden_spikes]

    # Read output spikes from register
    stat_reg_addr = 0x4B
    stat = read_register(stat_reg_addr)
    readout.output_spikes = [1 if stat & (1 << i) else 0 for i in range(output_count)]

    return readout

def apply_configuration():
    xylo_config = samna.xyloImu.configuration.XyloConfiguration()
    xylo_config.operation_mode = samna.xyloImu.OperationMode.Manual

    input_count = 3
    hidden_count = 5
    output_count = 2
    xylo_config.input.weights = [[1] * hidden_count] * input_count
    xylo_config.hidden.weights = [[1] * hidden_count] * hidden_count
    hidden_neurons = [samna.xyloImu.configuration.HiddenNeuron()] * hidden_count
    xylo_config.hidden.neurons = hidden_neurons
    output_neurons = [samna.xyloImu.configuration.OutputNeuron()] * output_count
    xylo_config.readout.neurons = output_neurons
    xylo_config.readout.weights = [[1] * output_count] * hidden_count

    dk.get_model().apply_configuration(xylo_config)
    return xylo_config, input_count, hidden_count, output_count

xylo_config, input_count, hidden_count, output_count = apply_configuration()  


def send_spikes(neurons):
    events = []
    for n in neurons:
        ev = samna.xyloImu.event.Spike()
        ev.neuron_id = n
        events.append(ev)
    source.write(events)

def evolve(input_neurons):
    send_spikes(input_neurons)
    trigger_processing()
    readout = request_readout(hidden_count, output_count)
    print("imformation after process：", readout)


readout = request_readout(hidden_count, output_count)
print("beginning imfromation：", readout)


evolve([])             
# evolve([1,1,2,2,2,0,0])     
# evolve([1,2,3])                 
# evolve([2,2,1,1,0])         

graph.stop()  
