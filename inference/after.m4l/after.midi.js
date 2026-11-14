// inlets & outlets
inlets = 2
outlets = 2
// Global current model
var cur_prior = 0;
var cur_encoder = 0;
var cur_decoder = 0;
var timbre_recv = 0;
var timbre_unp = 0;
var cur_z = 0;
// Instance prefix
var prefix_z = "";
var prefix_p = "";
// Object creation variable
var p = this.patcher

var cur_connect = []

// Set prefix
function prefix(val_z, val_p)
{
	prefix_z = val_z
	prefix_p = val_p
}

// Create a given model
function create(model_name, buffer_size, prior) 
{
	if (String(model_name).indexOf("off") !== -1)//(model_name == "off")
	{
		_delete_current()
		return
	}
	else 
	{
		_delete_current()	
	}
	// Replace current model
	if (cur_z != 0)
		_delete_current()

	// Create decoder
	var max_decoder = p.newdefault(100, 350, "nn~", model_name, "diffuse", buffer_size);
	
	var max_codec_decoder = p.newdefault(100, 600, "nn~", model_name, "decode", buffer_size);
	
	
	
	max_decoder.rect = [100, 350, 850, 25]
	max_codec_decoder.rect = [100, 600, 850, 25]
	
	// Number of latents
	var n_latents = 8; // The device has eight pitch/velocity voice pairs.
	var n_latents_timbre = Number(max_decoder.getboxattr('numinlets')) - 2 * n_latents;
	var n_audio_channels = Number(max_codec_decoder.getboxattr('numoutlets'));
	
	var n_ae_latents = Number(max_decoder.getboxattr('numoutlets'))

	// Gather all 
	cur_z = []
	// cur_encoder = max_encoder
	// cur_decoder = max_decoder

	cur_connect.push(max_decoder)
	cur_connect.push(max_codec_decoder)
	if ((n_audio_channels !== 1 && n_audio_channels !== 2) ||
		n_latents_timbre < 1 || n_latents_timbre > 8 ||
		Number(max_codec_decoder.getboxattr('numinlets')) !== n_ae_latents) {
		post("[ERROR] Incompatible MIDI generator/decoder channel layout\n");
		_delete_current();
		return;
	}
	var rave_out_r = p.getnamed("rave_out_r");
	var channel_mode = p.getnamed("audio_channels");
	if (n_audio_channels === 2 && (!rave_out_r || !channel_mode)) {
		post("[ERROR] This stereo model requires the updated MIDI device\n");
		_delete_current();
		return;
	}

	// Make major in / out connections
	p.connect(p.getnamed("model_props"), 0, max_decoder, 0)
	p.connect(p.getnamed("model_props"), 0, max_codec_decoder, 0)
	p.connect(p.getnamed("msg_in"), 0, max_decoder, 0)
	
	
	
	p.connect(max_codec_decoder, 0, p.getnamed("rave_out"), 0)
	// Legacy mono exports retain duplicated output; stereo keeps L/R separate.
	if (rave_out_r)
		p.connect(max_codec_decoder, n_audio_channels === 2 ? 1 : 0, rave_out_r, 0)

	var router = p.getnamed("midi_router")
	cur_prior = 0

	var main_plot = p.getnamed("main_plot");

	main_plot.setattr("bkgndpict", model_name + ".png")
	// Create prior

	// ### DIFFUSION MODEL ### 
	for (var z = 0; z < n_latents; z+=1)
	{
		// cur_connect = []

		var un_p = p.newdefault(100 + (100 * z), 180, "unpack", "0", "0");
		un_p.rect = [100 + (100 * z), 115, 185 + (100 * z), 115]

		var sig0 = p.newdefault(100 + (100 * z), 140,"sig~");
		sig0.rect = [100 + (100 * z), 150, 140 + (100 * z), 150]

		var sig1= p.newdefault(150 + (100 * z), 140,"sig~");
		sig1.rect = [150 + (100 * z), 150, 190 + (100 * z), 150]

		p.connect(router, z, un_p, 0)
		p.connect(un_p, 0, sig0, 0)
		p.connect(un_p, 1, sig1, 0)
		p.connect(sig0, 0, max_decoder, 2*z)
		p.connect(sig1, 0, max_decoder, 2*z+1)

		cur_connect.push(un_p)
		cur_connect.push(sig0)
		cur_connect.push(sig1)

	}
	
	
	//### TIMBRE CONTROL####
	var timbre_recv = p.getnamed("timbre_receive")
	var timbre_unp = p.newdefault(100 + (100 * z), 190, "mc.unpack~",n_latents_timbre);
	timbre_unp.rect = [100 + (100 * z), 190, 190 + (100 * z), 190]

	p.connect(timbre_recv, 0,timbre_unp,  0)

	for (var j = 0; j < n_latents_timbre; j++)
		{
			p.connect(timbre_unp, j,max_decoder,  n_latents*2 + j)
		}


	cur_connect.push(timbre_unp)

	
	
	
	//### AE LATENTS #### 
	for (var z = 0; z < n_ae_latents; z+=1)
	{
		//

	
		
		p.connect(max_decoder, z, max_codec_decoder, z);
		cur_z.push(cur_connect)
		}
		
	

	


	//### TIMBRE CONNECTION ###
	

	var sigm2l1 = p.getnamed("sigm2l1");
	var sigm2l2 = p.getnamed("sigm2l2");
	var packm2l = p.getnamed("packm2l");


	var unpackl2m = p.getnamed("unpackl2m");
	var snapl2m1= p.getnamed("snapl2m1");
	var snapl2m2 = p.getnamed("snapl2m2");

	var max_m2l =  p.newdefault(2567, 1639, "nn~", model_name, "map2latent");
	var max_l2m =  p.newdefault(3750, 2300, "nn~", model_name, "latent2map");
	cur_connect.push(max_m2l);
	cur_connect.push(max_l2m);
	if (Number(max_m2l.getboxattr('numinlets')) !== 2 ||
		Number(max_m2l.getboxattr('numoutlets')) !== n_latents_timbre ||
		Number(max_l2m.getboxattr('numinlets')) !== n_latents_timbre ||
		Number(max_l2m.getboxattr('numoutlets')) !== 2) {
		post("[ERROR] Learned map dimensions do not match MIDI timbre inputs\n");
		_delete_current();
		return;
	}


	p.connect(sigm2l1, 0 ,max_m2l,  0);
	p.connect(sigm2l2, 0 ,max_m2l,  1);

	for (var j = 0; j < n_latents_timbre; j++)
	{
		p.connect(max_m2l, j ,packm2l,  j)
	}
	
	for (var j = 0; j < n_latents_timbre; j++)
		{
			p.connect(unpackl2m, j ,max_l2m,  j)
		}
		
		p.connect(max_l2m, 0, snapl2m1, 0)
		p.connect(max_l2m, 1, snapl2m2, 0)

	if (channel_mode)
		channel_mode.message("int", n_audio_channels);
	outlet(1, "create")
	outlet(0, "bang")
}

function delete()
{
	if (cur_z != 0)
		_delete_current()
}

// delete current model
function _delete_current()
{	
	var channel_mode = p.getnamed("audio_channels");
	if (channel_mode)
		channel_mode.message("int", 1);

	for (var item in cur_connect){
		p.remove(cur_connect[item])
	}

	// p.remove(cur_decoder)
	// p.remove(timbre_recv)
	// p.remove(timbre_unp)

	var main_plot = p.getnamed("main_plot");
	main_plot.setattr("bkgndpict", "background_transparent.png")

	cur_connect = []
	// for (var t in cur_z)
	// {
	// 	var cur_c = cur_z[t]
	// 	for (var o in cur_c)
	// 		p.remove(cur_c[o])
	// }
	// cur_encoder = 0
	// cur_decoder = 0
	// cur_z = 0
}
