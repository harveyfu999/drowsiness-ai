package com.example.drowsinessdetector

import android.app.AlertDialog
import android.app.Dialog
import android.content.DialogInterface
import android.os.Bundle
import android.widget.EditText
import androidx.appcompat.app.AppCompatDialog
import androidx.appcompat.app.AppCompatDialogFragment

class EyeThresholdDialog(val userClick: (Float) -> Unit, val ele: Int, val msg: String) : AppCompatDialogFragment() {
    private lateinit var editEyeThreshold : EditText
    override fun onCreateDialog(savedInstanceState: Bundle?): Dialog {
        val builder = AlertDialog.Builder(activity)
        val view = activity?.layoutInflater?.inflate(R.layout.layout_dialog_eye_threshold, null)
        view?.let {
            builder.setView(view)
                .setTitle(msg)
                .setNegativeButton("cancel", object : DialogInterface.OnClickListener {
                    override fun onClick(p0: DialogInterface?, p1: Int) {
                    }
                })
                .setPositiveButton("ok", object : DialogInterface.OnClickListener {
                    override fun onClick(p0: DialogInterface?, p1: Int) {
                        editEyeThreshold.text.toString().toFloatOrNull()?.let(userClick)
                    }
                })
            editEyeThreshold = view.findViewById<EditText>(ele)
        }
        return builder.create()
    }
}